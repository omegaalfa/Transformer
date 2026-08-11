<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Backend\Contract;

use Omegaalfa\Transformer\Tensor\Tensor;

interface ElementWiseBackendInterface
{
    public function add(Tensor $left, Tensor $right): Tensor;
    public function sub(Tensor $left, Tensor $right): Tensor;
    public function mul(Tensor $left, Tensor $right): Tensor;
    public function div(Tensor $left, Tensor $right): Tensor;
}
