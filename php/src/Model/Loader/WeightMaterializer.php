<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Model\Loader;

use Omegaalfa\Transformer\Backend\BackendInterface;
use Omegaalfa\Transformer\Exception\SerializationException;
use Omegaalfa\Transformer\Serialization\Safetensors\SerializedTensor;
use Omegaalfa\Transformer\Tensor\DType;
use Omegaalfa\Transformer\Tensor\Tensor;

final readonly class WeightMaterializer
{
    public function __construct(private BackendInterface $backend)
    {
    }

    public function materialize(SerializedTensor $serialized, WeightMaterializationSpec $spec): Tensor
    {
        $metadata = $serialized->metadata;

        if ($metadata->dtype !== DType::Float32) {
            throw new SerializationException(sprintf(
                'Tensor "%s" uses %s; weight materialization currently accepts only Float32.',
                $metadata->name,
                $metadata->dtype->value,
            ));
        }

        if ($metadata->shape->dimensions !== $spec->checkpointShape->dimensions) {
            throw new SerializationException(sprintf('Tensor "%s" checkpoint shape does not match the expected shape.', $metadata->name));
        }

        if (strlen($serialized->bytes) !== $metadata->length) {
            throw new SerializationException(sprintf('Tensor "%s" payload length does not match its metadata.', $metadata->name));
        }

        $expectedLength = $this->checkedByteLength($metadata->shape->dimensions, $metadata->name);

        if ($metadata->length !== $expectedLength) {
            throw new SerializationException(sprintf('Tensor "%s" Float32 payload length does not match its shape.', $metadata->name));
        }

        $values = $this->decodeFloat32($serialized->bytes, $metadata->name);
        $values = match ($spec->orientation) {
            WeightOrientation::Identity => $values,
            WeightOrientation::PyTorchLinearTranspose => $this->transpose(
                $values,
                $spec->checkpointShape->dimensions[0],
                $spec->checkpointShape->dimensions[1],
            ),
        };

        return $this->backend->tensorFromFloat32($values, $spec->runtimeShape);
    }

    /**
     * @param list<int> $shape
     */
    private function checkedByteLength(array $shape, string $name): int
    {
        $elements = 1;

        foreach ($shape as $dimension) {
            if ($elements !== 0 && $dimension > intdiv(PHP_INT_MAX, $elements)) {
                throw new SerializationException(sprintf('Tensor "%s" element count overflows the platform integer size.', $name));
            }

            $elements *= $dimension;
        }

        if ($elements > intdiv(PHP_INT_MAX, 4)) {
            throw new SerializationException(sprintf('Tensor "%s" byte length overflows the platform integer size.', $name));
        }

        return $elements * 4;
    }

    /** @return list<float> */
    private function decodeFloat32(string $bytes, string $name): array
    {
        if ($bytes === '') {
            return [];
        }

        $decoded = unpack('g*', $bytes);

        if ($decoded === false) {
            throw new SerializationException(sprintf('Unable to decode tensor "%s" as little-endian Float32.', $name));
        }

        $values = [];

        foreach ($decoded as $value) {
            if (!is_float($value)) {
                throw new SerializationException(sprintf('Tensor "%s" produced an invalid decoded Float32 value.', $name));
            }

            if (!is_finite($value)) {
                throw new SerializationException(sprintf('Tensor "%s" contains a non-finite Float32 value.', $name));
            }

            $values[] = $value;
        }

        return $values;
    }

    /**
     * @param list<float> $values
     * @return list<float>
     */
    private function transpose(array $values, int $rows, int $columns): array
    {
        $transposed = [];

        for ($column = 0; $column < $columns; ++$column) {
            for ($row = 0; $row < $rows; ++$row) {
                $transposed[] = $values[$row * $columns + $column];
            }
        }

        return $transposed;
    }
}
