<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Backend;

use LogicException;
use Omegaalfa\Transformer\Tensor\Shape;
use Omegaalfa\Transformer\Tensor\Tensor;

abstract class AbstractBackend implements BackendInterface
{
    final public function matmul(Tensor $left, Tensor $right): Tensor
    {
        throw $this->notImplemented('matmul');
    }
    final public function transpose(Tensor $tensor, ?int $axisA = null, ?int $axisB = null): Tensor
    {
        throw $this->notImplemented('transpose');
    }
    final public function add(Tensor $left, Tensor $right): Tensor
    {
        throw $this->notImplemented('add');
    }
    final public function sub(Tensor $left, Tensor $right): Tensor
    {
        throw $this->notImplemented('sub');
    }
    final public function mul(Tensor $left, Tensor $right): Tensor
    {
        throw $this->notImplemented('mul');
    }
    final public function div(Tensor $left, Tensor $right): Tensor
    {
        throw $this->notImplemented('div');
    }
    final public function reshape(Tensor $tensor, Shape $shape): Tensor
    {
        throw $this->notImplemented('reshape');
    }
    final public function softmax(Tensor $tensor, int $axis = -1): Tensor
    {
        throw $this->notImplemented('softmax');
    }

    private function notImplemented(string $operation): LogicException
    {
        return new LogicException(sprintf('%s is not implemented by %s.', $operation, static::class));
    }
}
