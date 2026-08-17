<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tests\Integration\NN;

use InvalidArgumentException;
use Omegaalfa\Transformer\Backend\BackendType;
use Omegaalfa\Transformer\Backend\Ffi\FfiBackend;
use Omegaalfa\Transformer\Backend\Ffi\NativeLibrary;
use Omegaalfa\Transformer\Exception\BackendException;
use Omegaalfa\Transformer\NN\Activation\Gelu;
use Omegaalfa\Transformer\NN\LayerNorm;
use Omegaalfa\Transformer\NN\Linear;
use Omegaalfa\Transformer\NN\Module;
use Omegaalfa\Transformer\NN\Parameter;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Runtime\RuntimeConfig;
use Omegaalfa\Transformer\Tensor\Shape;
use Omegaalfa\Transformer\Tensor\Tensor;
use Omegaalfa\Transformer\Transformer\AttentionMask;
use Omegaalfa\Transformer\Transformer\Config\TransformerConfig;
use Omegaalfa\Transformer\Transformer\FeedForward;
use Omegaalfa\Transformer\Transformer\MultiHeadAttention;
use Omegaalfa\Transformer\Transformer\TransformerBlock;
use PHPUnit\Framework\TestCase;

final class TransformerBlockTest extends TestCase
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

    public function testMatchesIndependentPreNormReferenceWithAndWithoutMask(): void
    {
        $block = $this->block();
        $values = [1.5, -0.5, -1.0, 2.0, 0.25, 0.75];
        $input = $this->tensor($values, [1, 3, 2]);

        foreach ([null, new AttentionMask([true, false, true], new Shape([1, 3]))] as $mask) {
            $actual = $block->forward($input, $mask)->toFloat32();
            $expected = $this->blockReference($values, 1, 3, $mask?->values);
            $this->assertClose($expected, $actual);
        }
    }

    public function testExposesFrozenCompositionalIntrospectionTree(): void
    {
        $block = $this->block();
        self::assertInstanceOf(Module::class, $block);
        self::assertSame([], $block->parameters());
        self::assertSame(['norm1', 'attention', 'norm2', 'feed_forward'], array_keys($block->modules()));
        self::assertSame(
            ['input_projection', 'activation', 'output_projection'],
            array_keys($block->feedForward->modules()),
        );
        self::assertSame(['q_proj', 'k_proj', 'v_proj', 'out_proj'], array_keys($block->attention->modules()));
        self::assertSame(
            [
                'norm1.weight', 'norm1.bias',
                'attention.q_proj.weight', 'attention.k_proj.weight',
                'attention.v_proj.weight', 'attention.out_proj.weight',
                'norm2.weight', 'norm2.bias',
                'feed_forward.input_projection.weight', 'feed_forward.input_projection.bias',
                'feed_forward.output_projection.weight', 'feed_forward.output_projection.bias',
            ],
            array_keys($this->qualifiedParameters($block)),
        );
    }

    public function testPreservesAllApprovedEmptyShapes(): void
    {
        $block = $this->block();
        foreach ([[0, 3, 2], [2, 0, 2], [0, 0, 2]] as $shape) {
            $mask = new AttentionMask([], new Shape([$shape[0], $shape[1]]));
            $output = $block->forward($this->tensor([], $shape), $mask);
            self::assertSame($shape, $output->shape()->dimensions);
            self::assertSame([], $output->toFloat32());
        }
    }

    public function testRejectsConfigurationComponentsAndRuntimeMismatches(): void
    {
        $valid = $this->block();
        try {
            new TransformerBlock(
                new TransformerConfig(0, 1, 3, 1),
                $this->runtime,
                $valid->norm1,
                $valid->attention,
                $valid->norm2,
                $valid->feedForward,
            );
            self::fail('D=0 TransformerBlock accepted.');
        } catch (InvalidArgumentException) {
            self::addToAssertionCount(1);
        }

        $wrongNorm = $this->layerNorm(1, [1.0], [0.0]);
        try {
            new TransformerBlock($valid->config, $this->runtime, $wrongNorm, $valid->attention, $valid->norm2, $valid->feedForward);
            self::fail('Wrong normalization accepted.');
        } catch (InvalidArgumentException) {
            self::addToAssertionCount(1);
        }

        $foreignRuntime = $this->foreignRuntime();
        $foreignNorm = new LayerNorm(
            2,
            $foreignRuntime,
            new Parameter('weight', $this->foreignTensor([1.0, 1.0], [2], $foreignRuntime)),
            new Parameter('bias', $this->foreignTensor([0.0, 0.0], [2], $foreignRuntime)),
        );
        $this->expectException(InvalidArgumentException::class);
        new TransformerBlock($valid->config, $this->runtime, $foreignNorm, $valid->attention, $valid->norm2, $valid->feedForward);
    }

    public function testRejectsInputMaskAndFullyMaskedBatchThenRecovers(): void
    {
        $block = $this->block();
        $valid = $this->tensor([1.0, -1.0, 0.5, 0.25], [1, 2, 2]);
        $expected = $block->forward($valid)->toFloat32();

        foreach ([[[1.0, 2.0], [2]], [[1.0, 2.0], [1, 1, 2, 1]], [[1.0, 2.0, 3.0], [1, 1, 3]]] as [$values, $shape]) {
            try {
                $block->forward($this->tensor($values, $shape));
                self::fail('Invalid TransformerBlock input accepted.');
            } catch (InvalidArgumentException) {
                self::addToAssertionCount(1);
            }
        }

        try {
            $block->forward($valid, new AttentionMask([true, true], new Shape([2, 1])));
            self::fail('Incompatible mask accepted.');
        } catch (InvalidArgumentException) {
            self::addToAssertionCount(1);
        }
        try {
            $block->forward($valid, new AttentionMask([false, false], new Shape([1, 2])));
            self::fail('Fully masked batch accepted.');
        } catch (BackendException) {
            self::addToAssertionCount(1);
        }
        self::assertSame($expected, $block->forward($valid)->toFloat32());
    }

    public function testNonFiniteFailurePreservesInputMaskParametersAndLaterForward(): void
    {
        $block = $this->block();
        $valid = $this->tensor([1.0, -1.0, 0.5, 0.25], [1, 2, 2]);
        $mask = new AttentionMask([true, false], new Shape([1, 2]));
        $expected = $block->forward($valid, $mask)->toFloat32();
        $inputBefore = $valid->toFloat32();
        $maskBefore = $mask->values;
        $parameterBefore = $block->norm1->weight->tensor->toFloat32();

        foreach ([NAN, INF, -INF] as $invalid) {
            try {
                $block->forward($this->tensor([$invalid, 1.0], [1, 1, 2]));
                self::fail('Non-finite TransformerBlock input accepted.');
            } catch (BackendException) {
                self::addToAssertionCount(1);
            }
        }
        self::assertSame($expected, $block->forward($valid, $mask)->toFloat32());
        self::assertSame($inputBefore, $valid->toFloat32());
        self::assertSame($maskBefore, $mask->values);
        self::assertSame($parameterBefore, $block->norm1->weight->tensor->toFloat32());
    }

    public function testRepeatedForwardsOutputsAndInstancesAreIndependent(): void
    {
        $a = $this->block();
        $b = $this->block();
        $input = $this->tensor([1.0, -1.0, 0.5, 0.25], [1, 2, 2]);
        $first = $a->forward($input);
        $values = $first->toFloat32();
        $second = $a->forward($input);

        self::assertNotSame(spl_object_id($first->storage()), spl_object_id($second->storage()));
        self::assertSame($values, $second->toFloat32());
        self::assertSame($values, $b->forward($input)->toFloat32());
        self::assertSame($values, $first->toFloat32());

        $destroyed = $this->tensor([1.0, 2.0], [1, 1, 2]);
        $destroyed->destroy();
        $this->expectException(\LogicException::class);
        $a->forward($destroyed);
    }

    private function block(): TransformerBlock
    {
        $config = new TransformerConfig(2, 1, 3, 1);
        $identity = [1.0, 0.0, 0.0, 1.0];
        $attention = new MultiHeadAttention(
            $config,
            $this->runtime,
            $this->linear(2, 2, $identity, null),
            $this->linear(2, 2, $identity, null),
            $this->linear(2, 2, $identity, null),
            $this->linear(2, 2, $identity, null),
        );
        $feedForward = new FeedForward(
            $config,
            $this->runtime,
            $this->linear(2, 3, [0.5, -1.0, 0.25, 1.5, 0.75, -0.5], [0.2, -0.3, 0.4]),
            new Gelu($this->runtime),
            $this->linear(3, 2, [1.0, -0.5, 0.25, 0.75, -1.0, 0.5], [-0.1, 0.6]),
        );

        return new TransformerBlock(
            $config,
            $this->runtime,
            $this->layerNorm(2, [1.1, 0.9], [0.1, -0.2]),
            $attention,
            $this->layerNorm(2, [0.8, 1.2], [-0.1, 0.05]),
            $feedForward,
        );
    }

    /**
     * @param list<float> $weight
     * @param list<float>|null $bias
     */
    private function linear(int $input, int $output, array $weight, ?array $bias): Linear
    {
        return new Linear(
            $input,
            $output,
            $this->runtime,
            new Parameter('weight', $this->tensor($weight, [$input, $output])),
            $bias === null ? null : new Parameter('bias', $this->tensor($bias, [$output])),
        );
    }

    /**
     * @param list<float> $weight
     * @param list<float> $bias
     */
    private function layerNorm(int $d, array $weight, array $bias): LayerNorm
    {
        return new LayerNorm(
            $d,
            $this->runtime,
            new Parameter('weight', $this->tensor($weight, [$d])),
            new Parameter('bias', $this->tensor($bias, [$d])),
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
     * @param list<float> $values
     * @param list<int> $shape
     */
    private function foreignTensor(array $values, array $shape, Runtime $runtime): Tensor
    {
        $backend = $runtime->backend();
        if (!$backend instanceof FfiBackend) {
            throw new \LogicException('Test foreign runtime must use FFI.');
        }

        return $backend->tensorFromFloat32($values, new Shape($shape));
    }

    /** @return array<string, Parameter> */
    private function qualifiedParameters(Module $module, string $prefix = ''): array
    {
        $parameters = [];
        foreach ($module->parameters() as $name => $parameter) {
            $parameters[$prefix.$name] = $parameter;
        }
        foreach ($module->modules() as $name => $child) {
            foreach ($this->qualifiedParameters($child, $prefix.$name.'.') as $qualified => $parameter) {
                $parameters[$qualified] = $parameter;
            }
        }

        return $parameters;
    }

    /**
     * @param list<float|int> $input
     * @param list<bool>|null $mask
     * @return list<float>
     */
    private function blockReference(array $input, int $batch, int $sequence, ?array $mask): array
    {
        $norm1 = $this->layerNormReference($input, [1.1, 0.9], [0.1, -0.2], 2);
        $attention = $this->attentionReference($norm1, $batch, $sequence, $mask);
        $residual1 = [];
        foreach ($attention as $index => $value) {
            $residual1[] = $input[$index] + $value;
        }
        $norm2 = $this->layerNormReference($residual1, [0.8, 1.2], [-0.1, 0.05], 2);
        $hidden = $this->linearReference($norm2, [0.5, -1.0, 0.25, 1.5, 0.75, -0.5], [0.2, -0.3, 0.4], 2, 3);
        $activated = [];
        foreach ($hidden as $x) {
            $activated[] = 0.5 * $x * (1.0 + tanh(sqrt(2.0 / M_PI) * ($x + 0.044715 * $x * $x * $x)));
        }
        $ff = $this->linearReference($activated, [1.0, -0.5, 0.25, 0.75, -1.0, 0.5], [-0.1, 0.6], 3, 2);

        $output = [];
        foreach ($ff as $index => $value) {
            $output[] = $residual1[$index] + $value;
        }

        return $output;
    }

    /**
     * @param list<float|int> $input
     * @param list<float> $gamma
     * @param list<float> $beta
     * @return list<float>
     */
    private function layerNormReference(array $input, array $gamma, array $beta, int $d): array
    {
        $output = [];
        $rows = intdiv(count($input), $d);
        for ($row = 0; $row < $rows; ++$row) {
            $values = array_slice($input, $row * $d, $d);
            $mean = array_sum($values) / $d;
            $variance = 0.0;
            foreach ($values as $value) {
                $variance += ($value - $mean) ** 2;
            }
            $variance /= $d;
            for ($i = 0; $i < $d; ++$i) {
                $output[] = $gamma[$i] * ($values[$i] - $mean) / sqrt($variance + 1e-5) + $beta[$i];
            }
        }

        return $output;
    }

    /**
     * @param list<float> $input
     * @param list<bool>|null $mask
     * @return list<float>
     */
    private function attentionReference(array $input, int $batch, int $sequence, ?array $mask): array
    {
        $output = array_fill(0, count($input), 0.0);
        $scale = 1.0 / sqrt(2.0);
        for ($b = 0; $b < $batch; ++$b) {
            for ($query = 0; $query < $sequence; ++$query) {
                $scores = [];
                $maximum = -INF;
                for ($key = 0; $key < $sequence; ++$key) {
                    if ($mask !== null && !$mask[$b * $sequence + $key]) {
                        $scores[] = null;
                        continue;
                    }
                    $score = 0.0;
                    for ($inner = 0; $inner < 2; ++$inner) {
                        $score += $input[($b * $sequence + $query) * 2 + $inner]
                            * $input[($b * $sequence + $key) * 2 + $inner];
                    }
                    $scores[] = $score * $scale;
                    $maximum = max($maximum, $score * $scale);
                }
                $exponentials = array_map(static fn (?float $score): float => $score === null ? 0.0 : exp($score - $maximum), $scores);
                $sum = array_sum($exponentials);
                foreach ($exponentials as $key => $exponential) {
                    for ($inner = 0; $inner < 2; ++$inner) {
                        $output[($b * $sequence + $query) * 2 + $inner] += $exponential / $sum
                            * $input[($b * $sequence + $key) * 2 + $inner];
                    }
                }
            }
        }

        return array_values($output);
    }

    /**
     * @param list<float> $input
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
