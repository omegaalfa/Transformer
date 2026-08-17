<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tests\Integration\NN;

use InvalidArgumentException;
use Omegaalfa\Transformer\Backend\BackendType;
use Omegaalfa\Transformer\Backend\Ffi\FfiBackend;
use Omegaalfa\Transformer\Backend\Ffi\NativeLibrary;
use Omegaalfa\Transformer\Exception\BackendException;
use Omegaalfa\Transformer\NN\Activation\Gelu;
use Omegaalfa\Transformer\NN\Linear;
use Omegaalfa\Transformer\NN\Module;
use Omegaalfa\Transformer\NN\Parameter;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Runtime\RuntimeConfig;
use Omegaalfa\Transformer\Tensor\Shape;
use Omegaalfa\Transformer\Tensor\Tensor;
use Omegaalfa\Transformer\Transformer\Config\TransformerConfig;
use Omegaalfa\Transformer\Transformer\FeedForward;
use PHPUnit\Framework\TestCase;

final class FeedForwardTest extends TestCase
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

    public function testMatchesIndependentDToIToDReferenceAndAppliesBothBiases(): void
    {
        $inputWeight = [0.5, -1.0, 0.25, 1.5, 0.75, -0.5];
        $inputBias = [0.2, -0.3, 0.4];
        $outputWeight = [1.0, -0.5, 0.25, 0.75, -1.0, 0.5];
        $outputBias = [-0.1, 0.6];
        $feedForward = $this->feedForward($inputWeight, $inputBias, $outputWeight, $outputBias);
        $values = [-2.0, 1.0, 0.5, -1.5, 3.0, 2.0, -0.25, 0.75];

        $output = $feedForward->forward($this->tensor($values, [2, 2, 2]));
        $hidden = $this->linearReference($values, $inputWeight, $inputBias, 2, 3);
        $activated = array_map([$this, 'geluReference'], $hidden);
        $expected = $this->linearReference($activated, $outputWeight, $outputBias, 3, 2);

        self::assertSame([2, 2, 2], $output->shape()->dimensions);
        $this->assertClose($expected, $output->toFloat32());
    }

    public function testIsCompositionalModuleWithStableNamesAndResidentParameters(): void
    {
        $feedForward = $this->feedForward();
        self::assertInstanceOf(Module::class, $feedForward);
        self::assertSame([], $feedForward->parameters());
        self::assertSame(
            ['input_projection', 'activation', 'output_projection'],
            array_keys($feedForward->modules()),
        );
        self::assertSame(['weight', 'bias'], array_keys($feedForward->inputProjection->parameters()));
        self::assertSame([], $feedForward->activation->parameters());
        self::assertSame(['weight', 'bias'], array_keys($feedForward->outputProjection->parameters()));
    }

    public function testSupportsEveryLinearRankAndApprovedEmptyOuterShapes(): void
    {
        $feedForward = $this->feedForward();
        foreach ([[2], [2, 2], [1, 2, 2], [1, 1, 2, 2]] as $shape) {
            $numel = 1;
            foreach ($shape as $dimension) {
                $numel *= $dimension;
            }
            $output = $feedForward->forward($this->tensor(array_fill(0, $numel, 0.25), $shape));
            self::assertSame($shape, $output->shape()->dimensions);
        }
        foreach ([[0, 2], [2, 0, 2], [0, 0, 2]] as $shape) {
            $output = $feedForward->forward($this->tensor([], $shape));
            self::assertSame($shape, $output->shape()->dimensions);
            self::assertSame([], $output->toFloat32());
        }
    }

    public function testRejectsInvalidDimensionsProjectionsBiasAndRuntime(): void
    {
        $valid = $this->feedForward();
        $validInput = $valid->inputProjection;
        $validOutput = $valid->outputProjection;
        $gelu = $valid->activation;

        foreach ([new TransformerConfig(0, 1, 3, 1), new TransformerConfig(2, 1, 0, 1)] as $config) {
            try {
                new FeedForward($config, $this->runtime, $validInput, $gelu, $validOutput);
                self::fail('Invalid FeedForward dimensions accepted.');
            } catch (InvalidArgumentException) {
                self::addToAssertionCount(1);
            }
        }

        $withoutBias = new Linear(2, 3, $this->runtime, $validInput->weight);
        try {
            new FeedForward($valid->config, $this->runtime, $withoutBias, $gelu, $validOutput);
            self::fail('Biasless FeedForward projection accepted.');
        } catch (InvalidArgumentException) {
            self::addToAssertionCount(1);
        }

        $wrong = $this->linear(2, 2, [1, 0, 0, 1], [0, 0]);
        try {
            new FeedForward($valid->config, $this->runtime, $wrong, $gelu, $validOutput);
            self::fail('Wrong FeedForward projection accepted.');
        } catch (InvalidArgumentException) {
            self::addToAssertionCount(1);
        }

        $foreignRuntime = $this->foreignRuntime();
        $this->expectException(InvalidArgumentException::class);
        new FeedForward($valid->config, $this->runtime, $validInput, new Gelu($foreignRuntime), $validOutput);
    }

    public function testRejectsWrongInputAndNonFiniteThenRecovers(): void
    {
        $feedForward = $this->feedForward();
        $valid = $this->tensor([0.5, -0.25], [1, 1, 2]);
        $expected = $feedForward->forward($valid)->toFloat32();

        foreach ([[[1.0], []], [[1.0, 2.0, 3.0], [1, 1, 3]], [[NAN, 1.0], [1, 1, 2]], [[INF, 1.0], [1, 1, 2]], [[-INF, 1.0], [1, 1, 2]]] as [$values, $shape]) {
            try {
                $feedForward->forward($this->tensor($values, $shape));
                self::fail('Invalid FeedForward input accepted.');
            } catch (InvalidArgumentException|BackendException) {
                self::addToAssertionCount(1);
            }
        }
        self::assertSame($expected, $feedForward->forward($valid)->toFloat32());
    }

    public function testForwardsAndInstancesPreserveInputParametersAndOutputs(): void
    {
        $a = $this->feedForward();
        $b = $this->feedForward();
        $input = $this->tensor([0.5, -0.25], [1, 1, 2]);
        $inputBefore = $input->toFloat32();
        $weightBefore = $a->inputProjection->weight->tensor->toFloat32();
        $biasBefore = $a->outputProjection->bias?->tensor->toFloat32();
        $first = $a->forward($input);
        $values = $first->toFloat32();
        $second = $a->forward($input);

        self::assertNotSame(spl_object_id($first->storage()), spl_object_id($second->storage()));
        self::assertSame($values, $second->toFloat32());
        self::assertSame($values, $b->forward($input)->toFloat32());
        self::assertSame($values, $first->toFloat32());
        self::assertSame($inputBefore, $input->toFloat32());
        self::assertSame($weightBefore, $a->inputProjection->weight->tensor->toFloat32());
        self::assertSame($biasBefore, $a->outputProjection->bias?->tensor->toFloat32());
    }

    /**
     * @param list<float> $inputWeight
     * @param list<float> $inputBias
     * @param list<float> $outputWeight
     * @param list<float> $outputBias
     */
    private function feedForward(
        array $inputWeight = [1.0, 0.0, 0.5, 0.0, 1.0, -0.5],
        array $inputBias = [0.1, -0.2, 0.3],
        array $outputWeight = [1.0, 0.0, 0.0, 1.0, 0.5, -0.5],
        array $outputBias = [0.25, -0.25],
    ): FeedForward {
        $config = new TransformerConfig(2, 1, 3, 1);

        return new FeedForward(
            $config,
            $this->runtime,
            $this->linear(2, 3, $inputWeight, $inputBias),
            new Gelu($this->runtime),
            $this->linear(3, 2, $outputWeight, $outputBias),
        );
    }

    /**
     * @param list<float> $weight
     * @param list<float> $bias
     */
    private function linear(int $input, int $output, array $weight, array $bias): Linear
    {
        return new Linear(
            $input,
            $output,
            $this->runtime,
            new Parameter('weight', $this->tensor($weight, [$input, $output])),
            new Parameter('bias', $this->tensor($bias, [$output])),
        );
    }

    /**
     * @param list<float|int> $values
     * @param list<int> $shape
     */
    private function tensor(array $values, array $shape): Tensor
    {
        return $this->backend->tensorFromFloat32($values, new Shape($shape));
    }

    private function foreignRuntime(): Runtime
    {
        $path = NativeLibrary::defaultPath(dirname(__DIR__, 3));

        return new Runtime(new FfiBackend(new NativeLibrary($path)), new RuntimeConfig(BackendType::Ffi));
    }

    /**
     * @param list<float|int> $input
     * @param list<float> $weight
     * @param list<float> $bias
     * @return list<float>
     */
    private function linearReference(array $input, array $weight, array $bias, int $k, int $n): array
    {
        $output = [];
        $rows = intdiv(count($input), $k);
        for ($row = 0; $row < $rows; ++$row) {
            for ($column = 0; $column < $n; ++$column) {
                $value = $bias[$column];
                for ($inner = 0; $inner < $k; ++$inner) {
                    $value += $input[$row * $k + $inner] * $weight[$inner * $n + $column];
                }
                $output[] = $value;
            }
        }

        return $output;
    }

    private function geluReference(float $x): float
    {
        return 0.5 * $x * (1.0 + tanh(sqrt(2.0 / M_PI) * ($x + 0.044715 * $x * $x * $x)));
    }

    /**
     * @param list<float> $expected
     * @param list<float> $actual
     */
    private function assertClose(array $expected, array $actual): void
    {
        foreach ($expected as $index => $value) {
            self::assertLessThanOrEqual(1e-5 + 1e-5 * abs($value), abs($actual[$index] - $value));
        }
    }
}
