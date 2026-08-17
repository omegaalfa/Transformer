<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tests\Integration\NN;

use Omegaalfa\Transformer\Backend\BackendType;
use Omegaalfa\Transformer\Backend\Ffi\FfiBackend;
use Omegaalfa\Transformer\Backend\Ffi\NativeLibrary;
use Omegaalfa\Transformer\Exception\BackendException;
use Omegaalfa\Transformer\NN\LayerNorm;
use Omegaalfa\Transformer\NN\Parameter;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Runtime\RuntimeConfig;
use Omegaalfa\Transformer\Tensor\Shape;
use Omegaalfa\Transformer\Tensor\Tensor;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;

final class LayerNormTest extends TestCase
{
    private FfiBackend $backend;
    private Runtime $runtime;

    protected function setUp(): void
    {
        $path = NativeLibrary::defaultPath(dirname(__DIR__, 3));
        if (!is_file($path)) {
            self::markTestSkipped('Release native runtime is not built.');
        }
        $this->backend = new FfiBackend(new NativeLibrary($path));
        $this->runtime = new Runtime($this->backend, new RuntimeConfig(BackendType::Ffi));
    }

    /** @return iterable<string, array{list<int>, list<float|int>, float}> */
    public static function cases(): iterable
    {
        yield 'rank 1 D=1' => [[1], [42], 1e-5];
        yield 'rank 2 negative' => [[2, 3], [-3, -1, -2, 2, 0, 4], 1e-5];
        yield 'rank 3 constant' => [[2, 2, 2], [7, 7, 7, 7, 7, 7, 7, 7], 1e-4];
        yield 'rank N small and large' => [[1, 2, 1, 4], [1e-20, -1e-20, 1e20, -1e20, 3, -5, 8, 2], 1e-3];
    }

    /**
     * @param list<int>       $shape
     * @param list<float|int> $values
     */
    #[DataProvider('cases')]
    public function testMatchesIndependentPhpReference(array $shape, array $values, float $epsilon): void
    {
        $d = $shape[count($shape) - 1];
        $gamma = array_map(static fn (int $i): float => 0.25 + $i * 0.5, range(0, $d - 1));
        $beta = array_map(static fn (int $i): float => -0.75 + $i * 0.25, range(0, $d - 1));
        $layer = $this->layer($d, $gamma, $beta, $epsilon);
        $actual = $layer->forward($this->tensor($values, $shape));
        $expected = $this->reference($values, $gamma, $beta, $d, $epsilon);

        self::assertSame($shape, $actual->shape()->dimensions);
        foreach ($actual->toFloat32() as $i => $value) {
            self::assertLessThanOrEqual(1e-5 + 1e-5 * abs($expected[$i]), abs($value - $expected[$i]));
        }
    }

    public function testParametersAreResidentUnchangedAndOutputsIndependent(): void
    {
        $layer = $this->layer(3, [0.5, 2, -1], [1, -2, 3]);
        $weightBefore = $layer->weight->tensor->toFloat32();
        $biasBefore = $layer->bias->tensor->toFloat32();
        $input = $this->tensor([-2, 0, 2], [3]);
        $first = $layer->forward($input);
        $firstValues = $first->toFloat32();
        for ($i = 0; $i < 10; ++$i) {
            self::assertEqualsWithDelta($firstValues, $layer->forward($input)->toFloat32(), 0.0);
        }
        $second = $layer->forward($this->tensor([1, 1, 1], [3]));
        self::assertNotSame(spl_object_id($first->storage()), spl_object_id($second->storage()));
        self::assertSame($firstValues, $first->toFloat32());
        self::assertSame($weightBefore, $layer->weight->tensor->toFloat32());
        self::assertSame($biasBefore, $layer->bias->tensor->toFloat32());
        self::assertSame(['weight' => $layer->weight, 'bias' => $layer->bias], $layer->parameters());
    }

    public function testAllApprovedEmptyOuterShapes(): void
    {
        $layer = $this->layer(3, [1, 1, 1], [0, 0, 0]);
        foreach ([[0, 3], [2, 0, 3], [0, 0, 3]] as $shape) {
            $output = $layer->forward($this->tensor([], $shape));
            self::assertSame($shape, $output->shape()->dimensions);
            self::assertSame([], $output->toFloat32());
        }
    }

    public function testRejectsDimensionsParameterShapesAndEpsilon(): void
    {
        $valid = new Parameter('weight', $this->tensor([1], [1]));
        foreach ([0.0, -1.0, INF, NAN, 1e-999] as $epsilon) {
            try {
                new LayerNorm(1, $this->runtime, $valid, $valid, $epsilon);
                self::fail('Invalid epsilon accepted.');
            } catch (\InvalidArgumentException) {
            }
        }
        try {
            new LayerNorm(0, $this->runtime, $valid, $valid);
            self::fail('D=0 accepted.');
        } catch (\InvalidArgumentException) {
        }
        $wrong = new Parameter('wrong', $this->tensor([1, 1], [2]));
        $this->expectException(\InvalidArgumentException::class);
        new LayerNorm(1, $this->runtime, $wrong, $valid);
    }

    public function testValidInvalidValidAndNonFiniteRejection(): void
    {
        $layer = $this->layer(2, [1, 1], [0, 0]);
        $valid = $this->tensor([1, 2], [2]);
        $expected = $layer->forward($valid)->toFloat32();
        foreach ([[[1, 2, 3], [3]], [[NAN, 1], [2]], [[INF, 1], [2]], [[-INF, 1], [2]]] as [$values, $shape]) {
            try {
                $layer->forward($this->tensor($values, $shape));
                self::fail('Invalid input accepted.');
            } catch (\InvalidArgumentException|BackendException) {
            }
        }
        self::assertSame($expected, $layer->forward($valid)->toFloat32());
    }

    public function testRejectsNonFiniteGammaAndBeta(): void
    {
        $input = $this->tensor([1, 2], [2]);
        foreach ([[[NAN, 1], [0, 0]], [[1, 1], [0, INF]]] as [$gamma, $beta]) {
            $layer = $this->layer(2, $gamma, $beta);
            try {
                $layer->forward($input);
                self::fail('Non-finite parameter accepted.');
            } catch (BackendException) {
                self::addToAssertionCount(1);
            }
        }
    }

    public function testDestroyedAndForeignParametersFailWithoutAffectingIndependentInstance(): void
    {
        $a = $this->layer(2, [1, 1], [0, 0]);
        $b = $this->layer(2, [2, 3], [4, 5]);
        $input = $this->tensor([1, 2], [2]);
        $expectedB = $b->forward($input)->toFloat32();
        $a->weight->tensor->destroy();
        try {
            $a->forward($input);
            self::fail('Destroyed parameter accepted.');
        } catch (\LogicException) {
        }
        self::assertSame($expectedB, $b->forward($input)->toFloat32());

        $path = NativeLibrary::defaultPath(dirname(__DIR__, 3));
        $otherBackend = new FfiBackend(new NativeLibrary($path));
        $foreign = new Parameter('foreign', $otherBackend->tensorFromFloat32([1, 1], new Shape([2])));
        $this->expectException(\LogicException::class);
        new LayerNorm(2, $this->runtime, $foreign, $b->bias)->forward($input);
    }

    /**
     * @param list<float|int> $gamma
     * @param list<float|int> $beta
     */
    private function layer(int $d, array $gamma, array $beta, float $epsilon = 1e-5): LayerNorm
    {
        return new LayerNorm($d, $this->runtime, new Parameter('weight', $this->tensor($gamma, [$d])), new Parameter('bias', $this->tensor($beta, [$d])), $epsilon);
    }

    /**
     * @param list<float|int> $values
     * @param list<int>       $shape
     */
    private function tensor(array $values, array $shape): Tensor
    {
        return $this->backend->tensorFromFloat32($values, new Shape($shape));
    }

    /**
     * @param list<float|int> $input
     * @param list<float>     $gamma
     * @param list<float>     $beta
     * @return list<float>
     */
    private function reference(array $input, array $gamma, array $beta, int $d, float $epsilon): array
    {
        if ($d < 1) {
            throw new \InvalidArgumentException('Reference D must be positive.');
        }
        $output = [];
        foreach (array_chunk($input, $d) as $row) {
            $mean = array_sum($row) / $d;
            $variance = array_sum(array_map(static fn ($x): float => ($x - $mean) ** 2, $row)) / $d;
            for ($i = 0; $i < $d; ++$i) {
                $output[] = $gamma[$i] * ($row[$i] - $mean) / sqrt($variance + $epsilon) + $beta[$i];
            }
        }
        return $output;
    }
}
