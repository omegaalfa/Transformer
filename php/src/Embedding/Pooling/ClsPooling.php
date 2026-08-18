<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Embedding\Pooling;

use InvalidArgumentException;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Tensor\Shape;
use Omegaalfa\Transformer\Tensor\Tensor;

final readonly class ClsPooling implements PoolingInterface
{
    public function __construct(public Runtime $runtime)
    {
    }

    public function pool(Tensor $hiddenStates, ?Tensor $attentionMask = null): Tensor
    {
        $shape = $hiddenStates->shape()->dimensions;
        if (count($shape) !== 3 || $shape[1] <= 0 || $shape[2] <= 0) {
            throw new InvalidArgumentException('CLS pooling requires hidden states [B,S,D] with S and D greater than zero.');
        }
        [$batch, $sequence, $dimensions] = $shape;
        $hidden = $hiddenStates->toFloat32();
        if (count($hidden) !== $batch * $sequence * $dimensions) {
            throw new InvalidArgumentException('CLS-pooling Tensor length does not match its shape.');
        }
        $output = [];
        for ($row = 0; $row < $batch; ++$row) {
            $offset = $row * $sequence * $dimensions;
            for ($dimension = 0; $dimension < $dimensions; ++$dimension) {
                $value = $hidden[$offset + $dimension];
                if (!is_finite($value)) {
                    throw new InvalidArgumentException('CLS pooling requires finite CLS values.');
                }
                $output[] = $value;
            }
        }

        return $this->runtime->backend()->tensorFromFloat32($output, new Shape([$batch, $dimensions]));
    }
}
