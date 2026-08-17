<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tests\Integration\NN;

use InvalidArgumentException;
use Omegaalfa\Transformer\Backend\BackendType;
use Omegaalfa\Transformer\Backend\Ffi\FfiBackend;
use Omegaalfa\Transformer\Backend\Ffi\NativeLibrary;
use Omegaalfa\Transformer\Exception\BackendException;
use Omegaalfa\Transformer\NN\Linear;
use Omegaalfa\Transformer\NN\Module;
use Omegaalfa\Transformer\NN\Parameter;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Runtime\RuntimeConfig;
use Omegaalfa\Transformer\Tensor\Shape;
use Omegaalfa\Transformer\Tensor\Tensor;
use Omegaalfa\Transformer\Transformer\AttentionMask;
use Omegaalfa\Transformer\Transformer\Config\TransformerConfig;
use Omegaalfa\Transformer\Transformer\MultiHeadAttention;
use PHPUnit\Framework\TestCase;

final class MultiHeadAttentionTest extends TestCase
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

    public function testOwnsFourResidentBiasFreeLinearModulesInDeterministicOrder(): void
    {
        $attention = $this->attention(4, 2, $this->weights(4));

        self::assertInstanceOf(Module::class, $attention);
        self::assertSame([], $attention->parameters());
        self::assertSame(
            ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
            array_keys($attention->modules()),
        );
        foreach ($attention->modules() as $module) {
            self::assertInstanceOf(Linear::class, $module);
            self::assertSame(['weight'], array_keys($module->parameters()));
            self::assertNull($module->bias);
        }
    }

    public function testMatchesIndependentDoubleReferenceWithAndWithoutMask(): void
    {
        $weights = $this->weights(4);
        $values = [0.2, -0.4, 0.6, 1.0, -0.3, 0.8, -0.5, 0.7, 1.2, -0.9, 0.1, 0.4];
        $attention = $this->attention(4, 2, $weights);
        $input = $this->tensor($values, [1, 3, 4]);

        $maxAbsolute = 0.0;
        $maxRelative = 0.0;
        foreach ([null, new AttentionMask([true, false, true], new Shape([1, 3]))] as $mask) {
            $actual = $attention->forward($input, $mask)->toFloat32();
            $expected = $this->reference($values, 1, 3, 4, 2, $weights, $mask?->values);
            foreach ($expected as $index => $value) {
                $absolute = abs($actual[$index] - $value);
                $relative = $value == 0.0 ? 0.0 : $absolute / abs($value);
                $maxAbsolute = max($maxAbsolute, $absolute);
                $maxRelative = max($maxRelative, $relative);
                self::assertLessThanOrEqual(1e-5 + 1e-5 * abs($value), $absolute);
            }
        }
        self::assertLessThanOrEqual(1e-5, $maxAbsolute);
        self::assertLessThanOrEqual(1e-5, $maxRelative);
    }

    public function testSupportsHOneHEqualsDSequenceOneAndMultipleBatches(): void
    {
        foreach ([[1, 1, 4, 1], [1, 2, 4, 4], [2, 2, 4, 2]] as [$batch, $sequence, $d, $heads]) {
            $values = [];
            for ($i = 0; $i < $batch * $sequence * $d; ++$i) {
                $values[] = (($i % 9) - 4) / 5.0;
            }
            $weights = $this->weights($d);
            $actual = $this->attention($d, $heads, $weights)
                ->forward($this->tensor($values, [$batch, $sequence, $d]))
                ->toFloat32();
            $expected = $this->reference($values, $batch, $sequence, $d, $heads, $weights, null);
            foreach ($expected as $index => $value) {
                self::assertLessThanOrEqual(1e-5 + 1e-5 * abs($value), abs($actual[$index] - $value));
            }
        }
    }

    public function testAcceptsEveryApprovedEmptyShape(): void
    {
        $attention = $this->attention(4, 2, $this->weights(4));
        foreach ([[0, 3, 4], [2, 0, 4], [0, 0, 4]] as $shape) {
            $batch = $shape[0];
            $sequence = $shape[1];
            $mask = new AttentionMask([], new Shape([$batch, $sequence]));
            $output = $attention->forward($this->tensor([], $shape), $mask);
            self::assertSame($shape, $output->shape()->dimensions);
            self::assertSame([], $output->toFloat32());
        }
    }

    public function testRejectsInvalidConfigurationProjectionAndInputShapes(): void
    {
        $weights = $this->weights(4);
        $valid = $this->linear(4, $weights[0]);
        foreach ([[0, 1], [4, 0], [3, 2]] as [$d, $heads]) {
            try {
                new MultiHeadAttention(
                    new TransformerConfig($d, $heads, 8, 1),
                    $this->runtime,
                    $valid,
                    $valid,
                    $valid,
                    $valid,
                );
                self::fail('Invalid Attention dimensions accepted.');
            } catch (InvalidArgumentException) {
                self::addToAssertionCount(1);
            }
        }
        $invalidProjection = new Linear(4, 3, $this->runtime, new Parameter('weight', $this->tensor(array_fill(0, 12, 1), [4, 3])));
        $config = new TransformerConfig(4, 2, 8, 1);
        $this->expectException(InvalidArgumentException::class);
        new MultiHeadAttention($config, $this->runtime, $invalidProjection, $valid, $valid, $valid);
    }

    public function testRejectsWrongInputRankHiddenDimensionAndMaskShape(): void
    {
        $attention = $this->attention(4, 2, $this->weights(4));
        foreach ([[4], [1, 4], [1, 1, 1, 4], [1, 1, 3]] as $shape) {
            $numel = 1;
            foreach ($shape as $dimension) {
                $numel *= $dimension;
            }
            $values = array_fill(0, $numel, 0.5);
            try {
                $attention->forward($this->tensor($values, $shape));
                self::fail('Invalid Attention input shape accepted.');
            } catch (InvalidArgumentException) {
                self::addToAssertionCount(1);
            }
        }
        $this->expectException(InvalidArgumentException::class);
        $attention->forward(
            $this->tensor(array_fill(0, 8, 0.5), [1, 2, 4]),
            new AttentionMask([true, true], new Shape([2, 1])),
        );
    }

    public function testRejectsFullyMaskedRowsAndRecovers(): void
    {
        $attention = $this->attention(4, 2, $this->weights(4));
        $input = $this->tensor(array_fill(0, 16, 0.25), [2, 2, 4]);
        $validMask = new AttentionMask([true, false, true, true], new Shape([2, 2]));
        $expected = $attention->forward($input, $validMask)->toFloat32();
        try {
            $attention->forward($input, new AttentionMask([true, true, false, false], new Shape([2, 2])));
            self::fail('Fully masked batch row accepted.');
        } catch (BackendException) {
            self::addToAssertionCount(1);
        }
        self::assertSame($expected, $attention->forward($input, $validMask)->toFloat32());
    }

    public function testRejectsNonFiniteInputAndWeightsWithoutCorruptingState(): void
    {
        $weights = $this->weights(4);
        $attention = $this->attention(4, 2, $weights);
        $valid = $this->tensor(array_fill(0, 8, 0.25), [1, 2, 4]);
        $expected = $attention->forward($valid)->toFloat32();
        foreach ([NAN, INF, -INF] as $invalid) {
            $values = array_fill(0, 8, 0.25);
            $values[3] = $invalid;
            try {
                $attention->forward($this->tensor($values, [1, 2, 4]));
                self::fail('Non-finite input accepted.');
            } catch (BackendException) {
                self::addToAssertionCount(1);
            }
        }
        self::assertSame($expected, $attention->forward($valid)->toFloat32());

        $invalidWeights = $weights;
        $invalidWeights[0][0] = NAN;
        $invalidAttention = $this->attention(4, 2, $invalidWeights);
        $this->expectException(BackendException::class);
        $invalidAttention->forward($valid);
    }

    public function testMultipleForwardsOutputsInputsMasksWeightsAndInstancesAreIndependent(): void
    {
        $weights = $this->weights(4);
        $a = $this->attention(4, 2, $weights);
        $b = $this->attention(4, 2, $weights);
        $input = $this->tensor(array_fill(0, 8, 0.25), [1, 2, 4]);
        $mask = new AttentionMask([true, false], new Shape([1, 2]));
        $inputBefore = $input->toFloat32();
        $maskBefore = $mask->values;
        $weightBefore = $a->qProj->weight->tensor->toFloat32();
        $first = $a->forward($input, $mask);
        $firstValues = $first->toFloat32();
        $second = $a->forward($input, $mask);
        self::assertNotSame(spl_object_id($first->storage()), spl_object_id($second->storage()));
        self::assertSame($firstValues, $second->toFloat32());
        self::assertSame($firstValues, $b->forward($input, $mask)->toFloat32());
        self::assertSame($inputBefore, $input->toFloat32());
        self::assertSame($maskBefore, $mask->values);
        self::assertSame($weightBefore, $a->qProj->weight->tensor->toFloat32());
        self::assertSame($firstValues, $first->toFloat32());
    }

    public function testRejectsDestroyedInputAndForeignRuntime(): void
    {
        $attention = $this->attention(4, 2, $this->weights(4));
        $destroyed = $this->tensor(array_fill(0, 4, 0.25), [1, 1, 4]);
        $destroyed->destroy();
        try {
            $attention->forward($destroyed);
            self::fail('Destroyed input accepted.');
        } catch (\LogicException) {
            self::addToAssertionCount(1);
        }

        $path = NativeLibrary::defaultPath(dirname(__DIR__, 3));
        $foreignBackend = new FfiBackend(new NativeLibrary($path));
        $foreign = $foreignBackend->tensorFromFloat32(array_fill(0, 4, 0.25), new Shape([1, 1, 4]));
        $this->expectException(\LogicException::class);
        $attention->forward($foreign);
    }

    /** @param list<list<float>> $weights */
    private function attention(int $dimensions, int $heads, array $weights): MultiHeadAttention
    {
        $config = new TransformerConfig($dimensions, $heads, max(1, $dimensions * 2), 1);

        return new MultiHeadAttention(
            $config,
            $this->runtime,
            $this->linear($dimensions, $weights[0]),
            $this->linear($dimensions, $weights[1]),
            $this->linear($dimensions, $weights[2]),
            $this->linear($dimensions, $weights[3]),
        );
    }

    /** @param list<float> $weight */
    private function linear(int $dimensions, array $weight): Linear
    {
        return new Linear(
            $dimensions,
            $dimensions,
            $this->runtime,
            new Parameter('weight', $this->tensor($weight, [$dimensions, $dimensions])),
        );
    }

    /** @return list<list<float>> */
    private function weights(int $dimensions): array
    {
        $weights = [];
        for ($projection = 0; $projection < 4; ++$projection) {
            $weight = [];
            for ($row = 0; $row < $dimensions; ++$row) {
                for ($column = 0; $column < $dimensions; ++$column) {
                    $weight[] = $row === $column
                        ? 1.0 + $projection * 0.1
                        : (($row + $column + $projection) % 3 - 1) * 0.05;
                }
            }
            $weights[] = $weight;
        }

        return $weights;
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
     * @param list<list<float>> $weights
     * @param list<bool>|null $mask
     * @return list<float>
     */
    private function reference(
        array $input,
        int $batch,
        int $sequence,
        int $dimensions,
        int $heads,
        array $weights,
        ?array $mask,
    ): array {
        $q = $this->project($input, $weights[0], $dimensions);
        $k = $this->project($input, $weights[1], $dimensions);
        $v = $this->project($input, $weights[2], $dimensions);
        $headDimensions = intdiv($dimensions, $heads);
        $scale = 1.0 / sqrt((float) $headDimensions);
        $merged = array_fill(0, count($input), 0.0);
        for ($b = 0; $b < $batch; ++$b) {
            for ($h = 0; $h < $heads; ++$h) {
                for ($query = 0; $query < $sequence; ++$query) {
                    /** @var list<float|null> $scores */
                    $scores = [];
                    $maximum = -INF;
                    for ($key = 0; $key < $sequence; ++$key) {
                        if ($mask !== null && !$mask[$b * $sequence + $key]) {
                            $scores[] = null;
                            continue;
                        }
                        $score = 0.0;
                        for ($inner = 0; $inner < $headDimensions; ++$inner) {
                            $qIndex = ($b * $sequence + $query) * $dimensions + $h * $headDimensions + $inner;
                            $kIndex = ($b * $sequence + $key) * $dimensions + $h * $headDimensions + $inner;
                            $score += $q[$qIndex] * $k[$kIndex];
                        }
                        $scores[] = $score * $scale;
                        $maximum = max($maximum, $score * $scale);
                    }
                    $sum = 0.0;
                    foreach ($scores as $key => $score) {
                        if ($score === null) {
                            $scores[$key] = 0.0;
                        } else {
                            $scores[$key] = exp($score - $maximum);
                            $sum += $scores[$key];
                        }
                    }
                    foreach ($scores as $key => $score) {
                        $probability = $score / $sum;
                        for ($inner = 0; $inner < $headDimensions; ++$inner) {
                            $outputIndex = ($b * $sequence + $query) * $dimensions + $h * $headDimensions + $inner;
                            $vIndex = ($b * $sequence + $key) * $dimensions + $h * $headDimensions + $inner;
                            $merged[$outputIndex] += $probability * $v[$vIndex];
                        }
                    }
                }
            }
        }

        return $this->project(array_values($merged), $weights[3], $dimensions);
    }

    /**
     * @param list<float|int> $input
     * @param list<float> $weight
     * @return list<float>
     */
    private function project(array $input, array $weight, int $dimensions): array
    {
        $output = [];
        $rows = intdiv(count($input), $dimensions);
        for ($row = 0; $row < $rows; ++$row) {
            for ($column = 0; $column < $dimensions; ++$column) {
                $value = 0.0;
                for ($inner = 0; $inner < $dimensions; ++$inner) {
                    $value += $input[$row * $dimensions + $inner]
                        * $weight[$inner * $dimensions + $column];
                }
                $output[] = $value;
            }
        }

        return $output;
    }
}
