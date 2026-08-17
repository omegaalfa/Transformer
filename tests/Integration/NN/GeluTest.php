<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tests\Integration\NN;

use Omegaalfa\Transformer\Backend\BackendType;
use Omegaalfa\Transformer\Backend\Ffi\FfiBackend;
use Omegaalfa\Transformer\Backend\Ffi\NativeLibrary;
use Omegaalfa\Transformer\Exception\BackendException;
use Omegaalfa\Transformer\NN\Activation\ActivationInterface;
use Omegaalfa\Transformer\NN\Activation\Gelu;
use Omegaalfa\Transformer\NN\TensorModule;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Runtime\RuntimeConfig;
use Omegaalfa\Transformer\Tensor\Shape;
use Omegaalfa\Transformer\Tensor\Tensor;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;

final class GeluTest extends TestCase
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

    public function testIsAStatelessTensorActivation(): void
    {
        $gelu = new Gelu($this->runtime);
        self::assertInstanceOf(ActivationInterface::class, $gelu);
        self::assertInstanceOf(TensorModule::class, $gelu);
        self::assertSame([], $gelu->parameters());
        self::assertSame([], $gelu->modules());
    }

    /** @return iterable<string, array{list<int>, list<float|int>}> */
    public static function shapes(): iterable
    {
        yield 'rank zero' => [[], [1.0]];
        yield 'rank one' => [[5], [-3, -0.1, 0, 0.1, 3]];
        yield 'rank two' => [[2, 3], [-10, -2, -1, 1, 2, 10]];
        yield 'rank three' => [[1, 2, 2], [-1e-7, 1e-7, -5, 5]];
        yield 'rank N' => [[1, 2, 1, 3], [-20, -0.5, 0, 0.5, 7, 20]];
    }

    /**
     * @param list<int>       $shape
     * @param list<float|int> $values
     */
    #[DataProvider('shapes')]
    public function testPreservesEveryRankAndMatchesIndependentReference(array $shape, array $values): void
    {
        $input = $this->tensor($values, $shape);
        $before = $input->toFloat32();
        $output = (new Gelu($this->runtime))->forward($input);
        self::assertSame($shape, $output->shape()->dimensions);
        self::assertSame($before, $input->toFloat32());
        $this->assertParity($output->toFloat32(), $before);
    }

    public function testAcceptsAllEmptyShapes(): void
    {
        $gelu = new Gelu($this->runtime);
        foreach ([[0], [0, 3], [2, 0, 3], [0, 0, 3]] as $shape) {
            $output = $gelu->forward($this->tensor([], $shape));
            self::assertSame($shape, $output->shape()->dimensions);
            self::assertSame([], $output->toFloat32());
        }
    }

    public function testMultipleForwardsInputsOutputsAndInstancesAreIndependent(): void
    {
        $a = new Gelu($this->runtime);
        $b = new Gelu($this->runtime);
        $inputA = $this->tensor([-2, 0, 2], [3]);
        $inputB = $this->tensor([-1, 1, 3], [3]);
        $first = $a->forward($inputA);
        $firstValues = $first->toFloat32();
        $second = $a->forward($inputB);
        self::assertNotSame(spl_object_id($first->storage()), spl_object_id($second->storage()));
        self::assertNotSame($firstValues, $second->toFloat32());
        for ($i = 0; $i < 10; ++$i) {
            self::assertSame($firstValues, $b->forward($inputA)->toFloat32());
        }
        self::assertSame($firstValues, $first->toFloat32());
    }

    public function testRejectsEveryNonFiniteValueAndRecovers(): void
    {
        $gelu = new Gelu($this->runtime);
        $valid = $this->tensor([-1, 0, 1], [3]);
        $expected = $gelu->forward($valid)->toFloat32();
        foreach ([NAN, INF, -INF] as $invalid) {
            try {
                $gelu->forward($this->tensor([0, $invalid], [2]));
                self::fail('Non-finite GELU input accepted.');
            } catch (BackendException) {
                self::addToAssertionCount(1);
            }
        }
        self::assertSame($expected, $gelu->forward($valid)->toFloat32());
    }

    public function testRejectsForeignRuntimeBeforeNativeOperation(): void
    {
        $path = NativeLibrary::defaultPath(dirname(__DIR__, 3));
        $foreignBackend = new FfiBackend(new NativeLibrary($path));
        $foreign = $foreignBackend->tensorFromFloat32([1], new Shape([1]));
        $this->expectException(\LogicException::class);
        (new Gelu($this->runtime))->forward($foreign);
    }

    public function testUseAfterDestroyFailsAndAnotherInstanceRemainsValid(): void
    {
        $gelu = new Gelu($this->runtime);
        $destroyed = $this->tensor([1], [1]);
        $valid = $this->tensor([2], [1]);
        $expected = $gelu->forward($valid)->toFloat32();
        $destroyed->destroy();
        try {
            $gelu->forward($destroyed);
            self::fail('Destroyed input accepted.');
        } catch (\LogicException) {
            self::addToAssertionCount(1);
        }
        self::assertSame($expected, (new Gelu($this->runtime))->forward($valid)->toFloat32());
    }

    public function testDenseParityRecordsMaximumAbsoluteAndRelativeErrors(): void
    {
        $values = [];
        for ($i = -2000; $i <= 2000; ++$i) {
            $values[] = $i / 100.0;
        }
        $values = [...$values, -3.4028234663852886e38, -1e-30, 1e-30, 3.4028234663852886e38];
        $actual = (new Gelu($this->runtime))->forward($this->tensor($values, [count($values)]))->toFloat32();
        [$maxAbsolute, $maxRelative] = $this->assertParity($actual, $values);
        self::assertGreaterThanOrEqual(0.0, $maxAbsolute);
        self::assertGreaterThanOrEqual(0.0, $maxRelative);
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
     * @param list<float>     $actual
     * @param list<float|int> $input
     * @return array{float, float}
     */
    private function assertParity(array $actual, array $input): array
    {
        $maxAbsolute = 0.0;
        $maxRelative = 0.0;
        foreach ($input as $i => $value) {
            $x = (float) $value;
            $expected = 0.5 * $x * (1.0 + tanh(sqrt(2.0 / M_PI) * ($x + 0.044715 * $x * $x * $x)));
            $absolute = abs($actual[$i] - $expected);
            $relative = $expected == 0.0 ? 0.0 : $absolute / abs($expected);
            $maxAbsolute = max($maxAbsolute, $absolute);
            $maxRelative = max($maxRelative, $relative);
            self::assertLessThanOrEqual(1e-6 + 1e-6 * abs($expected), $absolute);
        }
        return [$maxAbsolute, $maxRelative];
    }
}
