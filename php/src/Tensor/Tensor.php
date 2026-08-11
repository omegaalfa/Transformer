<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tensor;

use LogicException;
use Omegaalfa\Transformer\Tensor\Storage\StorageInterface;

final class Tensor
{
    public function __construct(
        private readonly Shape $shape,
        private readonly StorageInterface $storage,
    ) {
    }

    /** @param array<array-key, mixed> $values */
    public static function fromArray(array $values, DType $dtype = DType::Float32): self
    {
        throw new LogicException('Tensor construction is not implemented yet.');
    }

    public static function zeros(Shape $shape, DType $dtype = DType::Float32, Device $device = Device::CPU): self
    {
        throw new LogicException('Tensor construction is not implemented yet.');
    }

    public static function ones(Shape $shape, DType $dtype = DType::Float32, Device $device = Device::CPU): self
    {
        throw new LogicException('Tensor construction is not implemented yet.');
    }

    public static function full(Shape $shape, int|float $value, DType $dtype = DType::Float32, Device $device = Device::CPU): self
    {
        throw new LogicException('Tensor construction is not implemented yet.');
    }

    public function shape(): Shape
    {
        return $this->shape;
    }

    public function storage(): StorageInterface
    {
        return $this->storage;
    }

    public function ndim(): int
    {
        throw new LogicException('Tensor metadata is not implemented yet.');
    }
    public function size(): int
    {
        throw new LogicException('Tensor metadata is not implemented yet.');
    }
    public function reshape(Shape $shape): self
    {
        throw new LogicException('Tensor reshape is not implemented yet.');
    }
    public function squeeze(?int $axis = null): self
    {
        throw new LogicException('Tensor squeeze is not implemented yet.');
    }
    public function unsqueeze(int $axis): self
    {
        throw new LogicException('Tensor unsqueeze is not implemented yet.');
    }
    public function get(int ...$indices): int|float
    {
        throw new LogicException('Tensor indexing is not implemented yet.');
    }
    public function slice(int $axis, int $start, ?int $length = null): self
    {
        throw new LogicException('Tensor slicing is not implemented yet.');
    }
    public function gather(int $axis, self $indices): self
    {
        throw new LogicException('Tensor gather is not implemented yet.');
    }
    public function add(self $other): self
    {
        throw new LogicException('Tensor addition is not implemented yet.');
    }
    public function sub(self $other): self
    {
        throw new LogicException('Tensor subtraction is not implemented yet.');
    }
    public function mul(self $other): self
    {
        throw new LogicException('Tensor multiplication is not implemented yet.');
    }
    public function div(self $other): self
    {
        throw new LogicException('Tensor division is not implemented yet.');
    }
    public function sum(?int $axis = null): self
    {
        throw new LogicException('Tensor sum is not implemented yet.');
    }
    public function mean(?int $axis = null): self
    {
        throw new LogicException('Tensor mean is not implemented yet.');
    }
    public function max(?int $axis = null): self
    {
        throw new LogicException('Tensor max is not implemented yet.');
    }
    public function variance(?int $axis = null): self
    {
        throw new LogicException('Tensor variance is not implemented yet.');
    }
    public function matmul(self $other): self
    {
        throw new LogicException('Tensor matmul is not implemented yet.');
    }
    public function transpose(?int $axisA = null, ?int $axisB = null): self
    {
        throw new LogicException('Tensor transpose is not implemented yet.');
    }
    public function exp(): self
    {
        throw new LogicException('Tensor exp is not implemented yet.');
    }
    public function sqrt(): self
    {
        throw new LogicException('Tensor sqrt is not implemented yet.');
    }
    public function softmax(int $axis = -1): self
    {
        throw new LogicException('Tensor softmax is not implemented yet.');
    }
    public function gelu(): self
    {
        throw new LogicException('Tensor GELU is not implemented yet.');
    }
}
