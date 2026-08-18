<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Embedding\Pooling;

use InvalidArgumentException;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Tensor\Shape;
use Omegaalfa\Transformer\Tensor\Tensor;

final readonly class MeanPooling implements PoolingInterface
{
    public function __construct(public Runtime $runtime)
    {
    }

    public function pool(Tensor $hiddenStates, ?Tensor $attentionMask = null): Tensor
    {
        $shape = $hiddenStates->shape()->dimensions;
        if (count($shape) !== 3 || $attentionMask === null) {
            throw new InvalidArgumentException('Masked mean pooling requires hidden states [B,S,D] and an attention mask.');
        }
        [$batch, $sequence, $dimensions] = $shape;
        if ($dimensions <= 0 || $attentionMask->shape()->dimensions !== [$batch, $sequence]) {
            throw new InvalidArgumentException('Mean-pooling mask must have shape [B,S].');
        }
        $hidden = $hiddenStates->toFloat32();
        $mask = $attentionMask->toFloat32();
        if (count($hidden) !== $batch * $sequence * $dimensions || count($mask) !== $batch * $sequence) {
            throw new InvalidArgumentException('Mean-pooling Tensor lengths do not match their shapes.');
        }
        $output = array_fill(0, $batch * $dimensions, 0.0);
        for ($row = 0; $row < $batch; ++$row) {
            $valid = 0;
            for ($token = 0; $token < $sequence; ++$token) {
                $maskValue = $mask[$row * $sequence + $token];
                if ($maskValue !== 0.0 && $maskValue !== 1.0) {
                    throw new InvalidArgumentException('Mean-pooling attention mask must contain only zero or one.');
                }
                if ($maskValue === 0.0) {
                    continue;
                }
                ++$valid;
                $hiddenOffset = ($row * $sequence + $token) * $dimensions;
                $outputOffset = $row * $dimensions;
                for ($dimension = 0; $dimension < $dimensions; ++$dimension) {
                    $value = $hidden[$hiddenOffset + $dimension];
                    if (!is_finite($value)) {
                        throw new InvalidArgumentException('Mean-pooling hidden states must be finite.');
                    }
                    $output[$outputOffset + $dimension] += $value;
                }
            }
            if ($valid === 0) {
                throw new InvalidArgumentException('Mean pooling requires at least one valid token in every batch row.');
            }
            $outputOffset = $row * $dimensions;
            for ($dimension = 0; $dimension < $dimensions; ++$dimension) {
                $output[$outputOffset + $dimension] /= $valid;
            }
        }

        return $this->runtime->backend()->tensorFromFloat32(array_values($output), new Shape([$batch, $dimensions]));
    }
}
