<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Backend\Contract;

use Omegaalfa\Transformer\Tensor\Tensor;

interface AlgebraBackendInterface
{
    public function matmul(Tensor $left, Tensor $right): Tensor;

    public function linear(Tensor $input, Tensor $weight, ?Tensor $bias = null): Tensor;

    public function transpose(Tensor $tensor, ?int $axisA = null, ?int $axisB = null): Tensor;
}
