<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Backend\Contract;

use Omegaalfa\Transformer\Tensor\Tensor;

interface NormalizationBackendInterface
{
    public function layerNorm(Tensor $input, Tensor $weight, Tensor $bias, float $epsilon = 1.0e-5): Tensor;
}
