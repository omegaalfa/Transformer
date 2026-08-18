<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Embedding\Normalization;

use InvalidArgumentException;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Tensor\Tensor;

final readonly class L2Normalizer
{
    public function __construct(public Runtime $runtime)
    {
    }

    public function normalize(Tensor $input): Tensor
    {
        $shape = $input->shape()->dimensions;
        if (count($shape) !== 2 || $shape[1] <= 0) {
            throw new InvalidArgumentException('L2 normalization requires shape [B,D] with D > 0.');
        }
        [$batch, $dimensions] = $shape;
        $values = $input->toFloat32();
        for ($row = 0; $row < $batch; ++$row) {
            $offset = $row * $dimensions;
            $sum = 0.0;
            for ($dimension = 0; $dimension < $dimensions; ++$dimension) {
                $value = $values[$offset + $dimension];
                if (!is_finite($value)) {
                    throw new InvalidArgumentException('L2 normalization requires finite values.');
                }
                $sum += $value * $value;
            }
            $norm = sqrt($sum);
            if (!is_finite($norm) || $norm === 0.0) {
                throw new InvalidArgumentException('L2 normalization rejects zero or non-finite row norms.');
            }
            for ($dimension = 0; $dimension < $dimensions; ++$dimension) {
                $values[$offset + $dimension] /= $norm;
            }
        }

        return $this->runtime->backend()->tensorFromFloat32(array_values($values), $input->shape());
    }
}
