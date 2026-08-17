<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tensor\Storage;

use FFI;
use FFI\CData;
use LogicException;
use Omegaalfa\Transformer\Exception\BackendException;
use Omegaalfa\Transformer\Tensor\Buffer\Float32Buffer;
use Omegaalfa\Transformer\Tensor\DType;
use Omegaalfa\Transformer\Tensor\Device;
use Omegaalfa\Transformer\Tensor\Shape;

final class NativeStorage implements StorageInterface
{
    private const STATUS_OK = 0;

    private mixed $handle;

    /** @internal Native handles must be created by NativeLibrary. */
    public function __construct(
        private readonly FFI $ffi,
        mixed $handle,
    ) {
        if (!$handle instanceof CData) {
            throw new LogicException('Native tensor requires an FFI handle.');
        }

        $this->handle = $handle;
    }

    public function __destruct()
    {
        $this->destroy();
    }

    public function dtype(): DType
    {
        $output = $this->ffi->new('int[1]');
        // @phpstan-ignore method.notFound (Native methods are defined by FFI::cdef().)
        $this->requireStatus($this->ffi->transformer_tensor_dtype($this->handle(), $output), 'dtype');
        $value = $output[0];
        if (!is_int($value)) {
            throw new BackendException('Native dtype metadata is invalid.');
        }

        return match ($value) {
            0 => DType::Float32,
            default => throw new BackendException("Unsupported native dtype {$value}.")
        };
    }

    public function device(): Device
    {
        return Device::CPU;
    }

    public function rank(): int
    {
        $output = $this->ffi->new('size_t[1]');
        // @phpstan-ignore method.notFound (Native methods are defined by FFI::cdef().)
        $this->requireStatus($this->ffi->transformer_tensor_rank($this->handle(), $output), 'rank');
        $value = $output[0];
        if (!is_int($value)) {
            throw new BackendException('Native rank metadata is invalid.');
        }

        return $value;
    }

    public function size(): int
    {
        $output = $this->ffi->new('size_t[1]');
        // @phpstan-ignore method.notFound (Native methods are defined by FFI::cdef().)
        $this->requireStatus($this->ffi->transformer_tensor_numel($this->handle(), $output), 'numel');
        $value = $output[0];
        if (!is_int($value)) {
            throw new BackendException('Native numel metadata is invalid.');
        }

        return $value;
    }

    public function shape(): Shape
    {
        $rank = $this->rank();
        if ($rank === 0) {
            return new Shape([]);
        }

        $output = $this->ffi->new("size_t[{$rank}]");
        // @phpstan-ignore method.notFound (Native methods are defined by FFI::cdef().)
        $this->requireStatus($this->ffi->transformer_tensor_shape($this->handle(), $output, $rank), 'shape');
        $dimensions = [];
        for ($axis = 0; $axis < $rank; ++$axis) {
            $dimension = $output[$axis];
            if (!is_int($dimension)) {
                throw new BackendException('Native shape metadata is invalid.');
            }
            $dimensions[] = $dimension;
        }

        return new Shape($dimensions);
    }

    /** @return list<float> */
    public function toFloat32(): array
    {
        $length = $this->size();
        if ($length === 0) {
            return [];
        }

        $buffer = $this->ffi->new("float[{$length}]");
        $this->requireStatus(
            // @phpstan-ignore method.notFound (Native methods are defined by FFI::cdef().)
            $this->ffi->transformer_tensor_copy_data_f32($this->handle(), $buffer, $length),
            'copy_data_f32',
        );
        $values = [];
        for ($index = 0; $index < $length; ++$index) {
            $value = $buffer[$index];
            // The CData allocation is float[N], so one exact type guard is
            // sufficient and also narrows the value for static analysis.
            if (!is_float($value)) {
                throw new BackendException('Native float buffer returned an invalid value.');
            }
            $values[] = $value;
        }

        return $values;
    }

    public function exportFloat32Buffer(): Float32Buffer
    {
        $length = $this->size();
        $buffer = $this->ffi->new('float[' . max(1, $length) . ']');
        $this->requireStatus(
            // @phpstan-ignore method.notFound (Native methods are defined by FFI::cdef().)
            $this->ffi->transformer_tensor_copy_data_f32($this->handle(), $buffer, $length),
            'export_float32_buffer',
        );

        return new Float32Buffer($this->ffi, $buffer, $length, $this->shape());
    }

    public function add(self $other): self
    {
        return $this->binary('transformer_tensor_add', $other);
    }

    public function matmul(self $other): self
    {
        return $this->binary('transformer_tensor_matmul', $other);
    }

    public function linear(self $weight, ?self $bias = null): self
    {
        if ($this->ffi !== $weight->ffi || ($bias !== null && $this->ffi !== $bias->ffi)) {
            throw new LogicException('Native tensors belong to different runtime instances.');
        }

        $output = $this->ffi->new('TransformerTensor *[1]');
        $biasHandle = $bias?->handle();
        $this->requireStatus(
            // @phpstan-ignore method.notFound (Native methods are defined by FFI::cdef().)
            $this->ffi->transformer_tensor_linear_last_dim(
                $this->handle(),
                $weight->handle(),
                $biasHandle,
                $output,
            ),
            'transformer_tensor_linear_last_dim',
        );

        return new self($this->ffi, $output[0]);
    }

    public function layerNorm(self $weight, self $bias, float $epsilon = 1.0e-5): self
    {
        if ($this->ffi !== $weight->ffi || $this->ffi !== $bias->ffi) {
            throw new LogicException('Native tensors belong to different runtime instances.');
        }

        $output = $this->ffi->new('TransformerTensor *[1]');
        $this->requireStatus(
            // @phpstan-ignore method.notFound (Native methods are defined by FFI::cdef().)
            $this->ffi->transformer_tensor_layer_norm(
                $this->handle(),
                $weight->handle(),
                $bias->handle(),
                $epsilon,
                $output,
            ),
            'transformer_tensor_layer_norm',
        );

        return new self($this->ffi, $output[0]);
    }

    /**
     * @internal Embedding IDs are copied to a temporary signed integer buffer.
     * @param list<int> $tokenIds
     */
    public function embeddingTokenIds(array $tokenIds, int $batch, int $sequence, FFI $runtimeFfi): self
    {
        if ($this->ffi !== $runtimeFfi) {
            throw new LogicException('Native tensor belongs to a different runtime instance.');
        }

        $tokenBuffer = $this->ffi->new('int64_t[' . max(1, count($tokenIds)) . ']');
        foreach ($tokenIds as $index => $tokenId) {
            // @phpstan-ignore offsetAccess.nonOffsetAccessible (FFI C array.)
            $tokenBuffer[$index] = $tokenId;
        }
        $output = $this->ffi->new('TransformerTensor *[1]');
        $this->requireStatus(
            // @phpstan-ignore method.notFound (Native methods are defined by FFI::cdef().)
            $this->ffi->transformer_tensor_embedding(
                $tokenBuffer,
                $batch,
                $sequence,
                $this->handle(),
                $output,
            ),
            'transformer_tensor_embedding',
        );

        return new self($this->ffi, $output[0]);
    }

    public function softmax(): self
    {
        return $this->unary('transformer_tensor_softmax');
    }

    public function softmaxLastDim(): self
    {
        return $this->unary('transformer_tensor_softmax_last_dim');
    }

    public function transpose(): self
    {
        return $this->unary('transformer_tensor_transpose');
    }

    public function destroy(): void
    {
        if ($this->handle === null) {
            return;
        }

        // @phpstan-ignore method.notFound (Native methods are defined by FFI::cdef().)
        $this->ffi->transformer_tensor_destroy($this->handle);
        $this->handle = null;
    }

    public function isDestroyed(): bool
    {
        return $this->handle === null;
    }

    private function unary(string $function): self
    {
        $output = $this->ffi->new('TransformerTensor *[1]');
        $this->requireStatus($this->ffi->{$function}($this->handle(), $output), $function);

        return new self($this->ffi, $output[0]);
    }

    private function binary(string $function, self $other): self
    {
        if ($this->ffi !== $other->ffi) {
            throw new LogicException('Native tensors belong to different runtime instances.');
        }

        $output = $this->ffi->new('TransformerTensor *[1]');
        $this->requireStatus($this->ffi->{$function}($this->handle(), $other->handle(), $output), $function);

        return new self($this->ffi, $output[0]);
    }

    private function handle(): CData
    {
        if (!$this->handle instanceof CData) {
            throw new LogicException('Native tensor has already been destroyed.');
        }

        return $this->handle;
    }

    private function requireStatus(mixed $status, string $operation): void
    {
        if (!is_int($status)) {
            throw new BackendException("Native tensor {$operation} returned an invalid status.");
        }

        if ($status !== self::STATUS_OK) {
            throw new BackendException("Native tensor {$operation} failed with status {$status}.");
        }
    }
}
