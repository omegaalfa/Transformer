<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tensor;

use LogicException;
use Omegaalfa\Transformer\Tensor\Buffer\Float32Buffer;
use Omegaalfa\Transformer\Tensor\Storage\NativeStorage;
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
        return $this->storage instanceof NativeStorage ? $this->storage->rank() : count($this->shape->dimensions);
    }
    public function size(): int
    {
        if ($this->storage instanceof NativeStorage) {
            return $this->storage->size();
        }

        $size = 1;
        foreach ($this->shape->dimensions as $dimension) {
            $size *= $dimension;
        }

        return $size;
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
        return $this->fromNativeBinary($other, 'add');
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
        return $this->fromNativeBinary($other, 'matmul');
    }
    public function transpose(?int $axisA = null, ?int $axisB = null): self
    {
        if ($axisA !== null || $axisB !== null) {
            throw new LogicException('Native transpose currently supports rank-2 default axes only.');
        }

        $storage = $this->nativeStorage()->transpose();

        return new self($storage->shape(), $storage);
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
        $rank = count($this->shape->dimensions);
        if ($rank < 1 || ($axis !== -1 && $axis !== $rank - 1)) {
            throw new LogicException('Native softmax supports only the last axis.');
        }

        $nativeStorage = $this->nativeStorage();
        $storage = $rank === 1 ? $nativeStorage->softmax() : $nativeStorage->softmaxLastDim();

        return new self($storage->shape(), $storage);
    }
    public function gelu(): self
    {
        throw new LogicException('Tensor GELU is not implemented yet.');
    }

    /** @return list<float> */
    public function toFloat32(): array
    {
        return $this->nativeStorage()->toFloat32();
    }

    /** Experimental contiguous export that avoids one PHP zval per element. */
    public function exportFloat32Buffer(): Float32Buffer
    {
        return $this->nativeStorage()->exportFloat32Buffer();
    }

    public function destroy(): void
    {
        $this->nativeStorage()->destroy();
    }

    private function nativeStorage(): NativeStorage
    {
        if (!$this->storage instanceof NativeStorage) {
            throw new LogicException('Operation requires native storage.');
        }

        return $this->storage;
    }

    private function fromNativeBinary(self $other, string $operation): self
    {
        $left = $this->nativeStorage();
        $right = $other->nativeStorage();
        $storage = $operation === 'add' ? $left->add($right) : $left->matmul($right);

        return new self($storage->shape(), $storage);
    }
}
