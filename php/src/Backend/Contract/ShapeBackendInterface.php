<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Backend\Contract;

use Omegaalfa\Transformer\Tensor\Shape;
use Omegaalfa\Transformer\Tensor\Tensor;

interface ShapeBackendInterface
{
    public function reshape(Tensor $tensor, Shape $shape): Tensor;
}
