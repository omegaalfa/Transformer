<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tests\Integration\NN;

use InvalidArgumentException;
use Omegaalfa\Transformer\Backend\BackendType;
use Omegaalfa\Transformer\Backend\Ffi\FfiBackend;
use Omegaalfa\Transformer\Backend\Ffi\NativeLibrary;
use Omegaalfa\Transformer\Exception\BackendException;
use Omegaalfa\Transformer\NN\Linear;
use Omegaalfa\Transformer\NN\Parameter;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Runtime\RuntimeConfig;
use Omegaalfa\Transformer\Tensor\Shape;
use Omegaalfa\Transformer\Tensor\Tensor;
use PHPUnit\Framework\TestCase;

final class LinearTest extends TestCase
{
    private FfiBackend $backend;
    private Runtime $runtime;

    protected function setUp(): void
    {
        $library = NativeLibrary::defaultPath(dirname(__DIR__, 3));
        if (!is_file($library)) {
            self::markTestSkipped('Release native runtime is not built.');
        }
        $this->backend = new FfiBackend(new NativeLibrary($library));
        $this->runtime = new Runtime($this->backend, new RuntimeConfig(BackendType::Ffi));
    }

    public function testConstructsWithResidentParametersAndIntrospectsThem(): void
    {
        $weight = new Parameter('weight', $this->tensor([1, 0, 0, 1], [2, 2]));
        $bias = new Parameter('bias', $this->tensor([0.5, -0.5], [2]));
        $linear = new Linear(2, 2, $this->runtime, $weight, $bias);

        self::assertSame(['weight' => $weight, 'bias' => $bias], $linear->parameters());
        self::assertSame([], $linear->modules());
    }

    public function testRejectsInvalidDimensionsAndParameterShapes(): void
    {
        $weight = new Parameter('weight', $this->tensor([1], [1, 1]));
        $this->expectException(InvalidArgumentException::class);
        new Linear(0, 1, $this->runtime, $weight);
    }

    public function testRejectsInvalidWeightShape(): void
    {
        $weight = new Parameter('weight', $this->tensor([1, 2, 3, 4], [1, 4]));
        $this->expectException(InvalidArgumentException::class);
        new Linear(2, 2, $this->runtime, $weight);
    }

    public function testRejectsInvalidBiasShape(): void
    {
        $weight = new Parameter('weight', $this->tensor([1, 0, 0, 1], [2, 2]));
        $bias = new Parameter('bias', $this->tensor([1, 2, 3], [3]));
        $this->expectException(InvalidArgumentException::class);
        new Linear(2, 2, $this->runtime, $weight, $bias);
    }

    public function testProjectsRanksOneThroughFourWithOptionalBias(): void
    {
        $weightValues = [1.0, -1.0, 2.0, 0.5, 3.0, 2.0];
        $biasValues = [0.25, -0.5, 1.0];
        $weight = new Parameter('weight', $this->tensor($weightValues, [2, 3]));
        $bias = new Parameter('bias', $this->tensor($biasValues, [3]));
        $withBias = new Linear(2, 3, $this->runtime, $weight, $bias);
        $withoutBias = new Linear(2, 3, $this->runtime, $weight);

        foreach ([[2], [2, 2], [2, 2, 2], [1, 2, 2, 2]] as $shape) {
            $numel = 1;
            foreach ($shape as $dimension) {
                $numel *= $dimension;
            }
            $rows = intdiv($numel, 2);
            $inputValues = [];
            for ($index = 0; $index < $rows * 2; ++$index) {
                $inputValues[] = ($index % 5) - 2.0;
            }
            $input = $this->tensor($inputValues, $shape);
            $expectedShape = [...array_slice($shape, 0, -1), 3];

            $actual = $withBias->forward($input);
            self::assertSame($expectedShape, $actual->shape()->dimensions);
            self::assertSame(
                $this->reference($inputValues, $weightValues, $biasValues, 2, 3),
                $actual->toFloat32(),
            );
            self::assertSame(
                $this->reference($inputValues, $weightValues, null, 2, 3),
                $withoutBias->forward($input)->toFloat32(),
            );
        }
    }

    public function testMultipleInputsKeepParametersAndOutputsIndependent(): void
    {
        $weight = new Parameter('weight', $this->tensor([1, 0, 0, 1], [2, 2]));
        $bias = new Parameter('bias', $this->tensor([0.5, -0.5], [2]));
        $linear = new Linear(2, 2, $this->runtime, $weight, $bias);
        $inputA = $this->tensor([1, 2], [2]);
        $inputB = $this->tensor([-3, 4], [2]);
        $inputBefore = $inputA->toFloat32();
        $weightStorageId = spl_object_id($weight->tensor->storage());
        $biasStorageId = spl_object_id($bias->tensor->storage());
        $first = $linear->forward($inputA);
        $firstValues = $first->toFloat32();
        $second = $linear->forward($inputB);

        for ($iteration = 0; $iteration < 10; ++$iteration) {
            self::assertSame($firstValues, $linear->forward($inputA)->toFloat32());
        }
        self::assertSame($firstValues, $first->toFloat32());
        self::assertNotSame($firstValues, $second->toFloat32());
        self::assertSame($inputBefore, $inputA->toFloat32());
        self::assertSame($weightStorageId, spl_object_id($weight->tensor->storage()));
        self::assertSame($biasStorageId, spl_object_id($bias->tensor->storage()));
    }

    public function testFailureDoesNotConsumeInputOrParameters(): void
    {
        $weight = new Parameter('weight', $this->tensor([1, 0, 0, 1], [2, 2]));
        $linear = new Linear(2, 2, $this->runtime, $weight);
        $valid = $this->tensor([1, 2], [2]);
        $expected = $linear->forward($valid)->toFloat32();

        try {
            $linear->forward($this->tensor([1, 2, 3], [3]));
            self::fail('Incompatible input features must fail.');
        } catch (BackendException) {
            self::assertSame([1.0, 0.0, 0.0, 1.0], $weight->tensor->toFloat32());
        }
        self::assertSame($expected, $linear->forward($valid)->toFloat32());
    }

    public function testEmptyPrefixDimensionProducesAnEmptyOutputWithPreservedShape(): void
    {
        $weight = new Parameter('weight', $this->tensor([1, 0, 0, 1], [2, 2]));
        $bias = new Parameter('bias', $this->tensor([0.5, -0.5], [2]));
        $linear = new Linear(2, 2, $this->runtime, $weight, $bias);
        $output = $linear->forward($this->tensor([], [0, 2]));

        self::assertSame([0, 2], $output->shape()->dimensions);
        self::assertSame([], $output->toFloat32());
    }

    public function testDestroyedWeightFailsCleanlyWithoutAffectingAnotherModule(): void
    {
        $weightA = new Parameter('weight', $this->tensor([1, 0, 0, 1], [2, 2]));
        $weightB = new Parameter('weight', $this->tensor([2, 0, 0, 2], [2, 2]));
        $linearA = new Linear(2, 2, $this->runtime, $weightA);
        $linearB = new Linear(2, 2, $this->runtime, $weightB);
        $input = $this->tensor([1, 2], [2]);
        $expectedB = $linearB->forward($input)->toFloat32();
        $weightA->tensor->destroy();

        try {
            $linearA->forward($input);
            self::fail('A destroyed resident weight must not be usable.');
        } catch (\LogicException) {
            self::assertSame($expectedB, $linearB->forward($input)->toFloat32());
        }
    }

    public function testIndependentModulesDoNotShareOrInvalidateParameters(): void
    {
        $weightA = new Parameter('weight', $this->tensor([1, 0, 0, 1], [2, 2]));
        $weightB = new Parameter('weight', $this->tensor([2, 0, 0, 2], [2, 2]));
        $linearA = new Linear(2, 2, $this->runtime, $weightA);
        $linearB = new Linear(2, 2, $this->runtime, $weightB);
        $input = $this->tensor([1, 2], [2]);
        self::assertNotSame(spl_object_id($weightA->tensor->storage()), spl_object_id($weightB->tensor->storage()));
        self::assertNotSame($linearA->forward($input)->toFloat32(), $linearB->forward($input)->toFloat32());
        $expectedB = $linearB->forward($input)->toFloat32();
        unset($linearA, $weightA);

        self::assertSame($expectedB, $linearB->forward($input)->toFloat32());
    }

    /**
     * @param list<float|int> $values
     * @param list<int> $shape
     */
    private function tensor(array $values, array $shape): Tensor
    {
        return $this->backend->tensorFromFloat32($values, new Shape($shape));
    }

    /**
     * @param list<float|int> $input
     * @param list<float> $weight
     * @param list<float>|null $bias
     * @return list<float>
     */
    private function reference(
        array $input,
        array $weight,
        ?array $bias,
        int $inputFeatures,
        int $outputFeatures,
    ): array {
        if ($inputFeatures < 1) {
            throw new InvalidArgumentException('Reference input features must be positive.');
        }
        $output = [];
        foreach (array_chunk($input, $inputFeatures) as $row) {
            for ($column = 0; $column < $outputFeatures; ++$column) {
                $value = $bias[$column] ?? 0.0;
                for ($inner = 0; $inner < $inputFeatures; ++$inner) {
                    $value += $row[$inner] * $weight[$inner * $outputFeatures + $column];
                }
                $output[] = (float) $value;
            }
        }

        return $output;
    }
}
