<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Backend\Contract;

use Omegaalfa\Transformer\Tensor\Tensor;

interface ActivationBackendInterface
{
    public function gelu(Tensor $input): Tensor;

    public function softmax(Tensor $tensor, int $axis = -1): Tensor;
}
