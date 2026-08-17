<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Backend;

use LogicException;
use Omegaalfa\Transformer\Tensor\Shape;
use Omegaalfa\Transformer\Tensor\Tensor;
use Omegaalfa\Transformer\Transformer\AttentionMask;

abstract class AbstractBackend implements BackendInterface
{
    public function matmul(Tensor $left, Tensor $right): Tensor
    {
        throw $this->notImplemented('matmul');
    }
    public function linear(Tensor $input, Tensor $weight, ?Tensor $bias = null): Tensor
    {
        throw $this->notImplemented('linear');
    }
    public function embeddingTokenIds(array $tokenIds, Shape $shape, Tensor $weight): Tensor
    {
        throw $this->notImplemented('embeddingTokenIds');
    }
    public function layerNorm(Tensor $input, Tensor $weight, Tensor $bias, float $epsilon = 1.0e-5): Tensor
    {
        throw $this->notImplemented('layerNorm');
    }
    public function transpose(Tensor $tensor, ?int $axisA = null, ?int $axisB = null): Tensor
    {
        throw $this->notImplemented('transpose');
    }
    public function add(Tensor $left, Tensor $right): Tensor
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
    public function softmax(Tensor $tensor, int $axis = -1): Tensor
    {
        throw $this->notImplemented('softmax');
    }
    public function gelu(Tensor $input): Tensor
    {
        throw $this->notImplemented('gelu');
    }
    public function multiHeadAttention(
        Tensor $input,
        Tensor $qWeight,
        Tensor $kWeight,
        Tensor $vWeight,
        Tensor $outWeight,
        int $heads,
        ?AttentionMask $mask = null,
    ): Tensor {
        throw $this->notImplemented('multiHeadAttention');
    }

    private function notImplemented(string $operation): LogicException
    {
        return new LogicException(sprintf('%s is not implemented by %s.', $operation, static::class));
    }
}
