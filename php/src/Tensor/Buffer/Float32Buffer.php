<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tensor\Buffer;

use FFI;
use FFI\CData;
use LogicException;
use OutOfBoundsException;
use Omegaalfa\Transformer\Tensor\DType;
use Omegaalfa\Transformer\Tensor\Shape;

/**
 * Experimental owned contiguous Float32 export buffer.
 *
 * The underlying CData is deliberately not exposed. This buffer owns an
 * independent copy and therefore does not borrow the source Tensor lifetime.
 */
final class Float32Buffer
{
    private mixed $data;

    /** @internal Buffers must be created by NativeStorage. */
    public function __construct(
        private readonly FFI $ffi,
        mixed $data,
        private readonly int $numel,
        private readonly Shape $shape,
    ) {
        if (!$data instanceof CData || $numel < 0) {
            throw new LogicException('Float32 buffer requires valid owned CData.');
        }

        $this->data = $data;
    }

    public function __destruct()
    {
        $this->destroy();
    }

    public function numel(): int
    {
        $this->data();

        return $this->numel;
    }

    public function sizeBytes(): int
    {
        $this->data();
        if ($this->numel > intdiv(PHP_INT_MAX, 4)) {
            throw new LogicException('Float32 buffer byte length exceeds the platform integer size.');
        }

        return $this->numel * 4;
    }

    public function shape(): Shape
    {
        $this->data();

        return $this->shape;
    }

    public function dtype(): DType
    {
        $this->data();

        return DType::Float32;
    }

    public function valueAt(int $index): float
    {
        if ($index < 0 || $index >= $this->numel) {
            throw new OutOfBoundsException("Float32 buffer index {$index} is out of bounds.");
        }

        $value = $this->data()[$index];
        if (!is_float($value)) {
            throw new LogicException('Float32 CData returned an invalid value.');
        }

        return $value;
    }

    /** Returns an independent IEEE 754 Float32 little-endian byte string. */
    public function toBytes(): string
    {
        if ($this->numel === 0) {
            $this->data();

            return '';
        }

        $data = $this->data();
        $bytes = FFI::string($this->ffi->cast('char *', FFI::addr($data)), max(0, $this->sizeBytes()));
        if (pack('S', 1) === "\x01\x00") {
            return $bytes;
        }

        $littleEndian = '';
        foreach (str_split($bytes, 4) as $floatBytes) {
            $littleEndian .= strrev($floatBytes);
        }

        return $littleEndian;
    }

    public function destroy(): void
    {
        $this->data = null;
    }

    public function isDestroyed(): bool
    {
        return !$this->data instanceof CData;
    }

    private function data(): CData
    {
        if (!$this->data instanceof CData) {
            throw new LogicException('Float32 buffer has already been destroyed.');
        }

        return $this->data;
    }
}
