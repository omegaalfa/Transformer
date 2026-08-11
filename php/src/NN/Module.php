<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\NN;

use Omegaalfa\Transformer\Tensor\Tensor;

interface Module
{
    public function forward(Tensor $input): Tensor;
}
