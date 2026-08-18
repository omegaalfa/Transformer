<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Serialization\Safetensors;

use JsonException;
use Omegaalfa\Transformer\Exception\SerializationException;
use Omegaalfa\Transformer\Tensor\DType;
use Omegaalfa\Transformer\Tensor\Shape;

final class SafetensorsReader implements SafetensorsReaderInterface
{
    private const int PREFIX_BYTES = 8;
    private const int MAX_HEADER_BYTES = 100_000_000;

    public function open(string $path): SafetensorsReadSession
    {
        $handle = @fopen($path, 'rb');
        if ($handle === false) {
            throw new SerializationException(sprintf('Unable to open Safetensors file: %s', $path));
        }

        try {
            return new SafetensorsReadSession($handle, $this->inspect($handle, $path));
        } catch (\Throwable $exception) {
            fclose($handle);
            throw $exception;
        }
    }

    public function metadata(string $path): WeightMap
    {
        $handle = @fopen($path, 'rb');

        if ($handle === false) {
            throw new SerializationException(sprintf('Unable to open Safetensors file: %s', $path));
        }

        try {
            return $this->inspect($handle, $path);
        } finally {
            fclose($handle);
        }
    }

    public function tensor(string $path, string $name): SerializedTensor
    {
        if ($name === '') {
            throw new SerializationException('Safetensors tensor name must not be empty.');
        }

        $handle = @fopen($path, 'rb');

        if ($handle === false) {
            throw new SerializationException(sprintf('Unable to open Safetensors file: %s', $path));
        }

        try {
            $weightMap = $this->inspect($handle, $path);
            $metadata = $weightMap->tensors[$name] ?? null;

            if ($metadata === null) {
                throw new SerializationException(sprintf('Safetensors tensor "%s" does not exist.', $name));
            }

            if (fseek($handle, $metadata->offset, SEEK_SET) !== 0) {
                throw new SerializationException(sprintf('Unable to seek to Safetensors tensor "%s".', $name));
            }

            return new SerializedTensor(
                metadata: $metadata,
                bytes: $this->readExactly($handle, $metadata->length, sprintf('tensor "%s" payload', $name)),
            );
        } finally {
            fclose($handle);
        }
    }

    /** @param resource $handle */
    private function inspect($handle, string $path): WeightMap
    {
        $fileSize = $this->fileSize($handle, $path);
        $prefix = $this->readExactly($handle, self::PREFIX_BYTES, 'header length');
        $headerLength = $this->decodeHeaderLength($prefix);

        if ($headerLength === 0 || $headerLength > self::MAX_HEADER_BYTES) {
            throw new SerializationException(sprintf(
                'Invalid Safetensors header length: %d bytes.',
                $headerLength,
            ));
        }

        $dataStart = $this->checkedAdd(self::PREFIX_BYTES, $headerLength, 'header boundary');

        if ($dataStart > $fileSize) {
            throw new SerializationException('Safetensors header extends beyond the end of the file.');
        }

        $header = $this->readExactly($handle, $headerLength, 'header');
        $decoded = $this->decodeHeader($header);

        return new WeightMap($this->parseTensors($decoded, $dataStart, $fileSize));
    }

    /** @param resource $handle */
    private function fileSize($handle, string $path): int
    {
        $stat = fstat($handle);

        if ($stat === false) {
            throw new SerializationException(sprintf('Unable to determine Safetensors file size: %s', $path));
        }

        return $stat['size'];
    }

    /** @param resource $handle */
    private function readExactly($handle, int $length, string $section): string
    {
        $contents = '';

        while (strlen($contents) < $length) {
            $remaining = $length - strlen($contents);

            if ($remaining < 1) {
                break;
            }

            $chunk = fread($handle, $remaining);

            if ($chunk === false || $chunk === '') {
                throw new SerializationException(sprintf('Truncated Safetensors %s.', $section));
            }

            $contents .= $chunk;
        }

        return $contents;
    }

    private function decodeHeaderLength(string $prefix): int
    {
        /** @var array{low: int, high: int}|false $parts */
        $parts = unpack('Vlow/Vhigh', $prefix);

        if ($parts === false) {
            throw new SerializationException('Unable to decode Safetensors header length.');
        }

        if ($parts['high'] > 0x7FFFFFFF) {
            throw new SerializationException('Safetensors header length exceeds the platform integer range.');
        }

        return $this->checkedAdd(
            $parts['low'],
            $parts['high'] * 4_294_967_296,
            'header length',
        );
    }

    /** @return array<string, mixed> */
    private function decodeHeader(string $header): array
    {
        try {
            $decoded = json_decode($header, true, 512, JSON_THROW_ON_ERROR);
        } catch (JsonException $exception) {
            throw new SerializationException(
                sprintf('Invalid Safetensors header JSON: %s', $exception->getMessage()),
                previous: $exception,
            );
        }

        if (!is_array($decoded) || array_is_list($decoded)) {
            throw new SerializationException('Safetensors header must be a JSON object.');
        }

        $headerObject = [];

        foreach ($decoded as $name => $entry) {
            if (!is_string($name)) {
                throw new SerializationException('Safetensors header object keys must be strings.');
            }

            $headerObject[$name] = $entry;
        }

        return $headerObject;
    }

    /**
     * @param array<string, mixed> $header
     * @return array<string, TensorMetadata>
     */
    private function parseTensors(array $header, int $dataStart, int $fileSize): array
    {
        $tensors = [];
        $ranges = [];

        foreach ($header as $name => $entry) {
            if ($name === '__metadata__') {
                $this->validateUserMetadata($entry);

                continue;
            }

            if ($name === '') {
                throw new SerializationException('Safetensors tensor names must not be empty.');
            }

            if (!is_array($entry) || array_is_list($entry)) {
                throw new SerializationException(sprintf('Metadata for tensor "%s" must be an object.', $name));
            }

            $dtype = $this->parseDType($entry['dtype'] ?? null, $name);
            $shape = $this->parseShape($entry['shape'] ?? null, $name);
            [$relativeStart, $relativeEnd] = $this->parseOffsets($entry['data_offsets'] ?? null, $name);
            $length = $relativeEnd - $relativeStart;
            $expectedLength = $this->expectedByteLength($shape, $dtype, $name);

            if ($length !== $expectedLength) {
                throw new SerializationException(sprintf(
                    'Tensor "%s" data length is %d bytes; expected %d from dtype and shape.',
                    $name,
                    $length,
                    $expectedLength,
                ));
            }

            $absoluteStart = $this->checkedAdd($dataStart, $relativeStart, sprintf('offset for tensor "%s"', $name));
            $absoluteEnd = $this->checkedAdd($dataStart, $relativeEnd, sprintf('end offset for tensor "%s"', $name));

            if ($absoluteEnd > $fileSize) {
                throw new SerializationException(sprintf('Tensor "%s" extends beyond the end of the file.', $name));
            }

            $tensors[$name] = new TensorMetadata(
                name: $name,
                shape: new Shape($shape),
                dtype: $dtype,
                offset: $absoluteStart,
                length: $length,
            );
            $ranges[] = [$relativeStart, $relativeEnd, $name];
        }

        $this->validatePayloadCoverage($ranges, $fileSize - $dataStart);

        return $tensors;
    }

    /** @param list<array{int, int, string}> $ranges */
    private function validatePayloadCoverage(array $ranges, int $payloadLength): void
    {
        usort($ranges, static function (array $left, array $right): int {
            return [$left[0], $left[1], $left[2]] <=> [$right[0], $right[1], $right[2]];
        });

        $cursor = 0;

        foreach ($ranges as [$start, $end, $name]) {
            if ($start !== $cursor) {
                $problem = $start < $cursor ? 'overlaps another tensor' : 'leaves an unindexed gap';

                throw new SerializationException(sprintf('Tensor "%s" %s in the Safetensors payload.', $name, $problem));
            }

            $cursor = $end;
        }

        if ($cursor !== $payloadLength) {
            throw new SerializationException('Safetensors payload contains unindexed trailing bytes.');
        }
    }

    private function validateUserMetadata(mixed $metadata): void
    {
        if (!is_array($metadata) || array_is_list($metadata)) {
            throw new SerializationException('Safetensors __metadata__ must be an object.');
        }

        foreach ($metadata as $key => $value) {
            if (!is_string($key) || !is_string($value)) {
                throw new SerializationException('Safetensors __metadata__ keys and values must be strings.');
            }
        }
    }

    private function parseDType(mixed $value, string $name): DType
    {
        if (!is_string($value)) {
            throw new SerializationException(sprintf('Tensor "%s" has no valid dtype.', $name));
        }

        return match ($value) {
            'F32' => DType::Float32,
            'F16' => DType::Float16,
            'BF16' => DType::BFloat16,
            'I64' => DType::Int64,
            'I8' => DType::Int8,
            default => throw new SerializationException(sprintf(
                'Tensor "%s" uses unsupported Safetensors dtype "%s".',
                $name,
                $value,
            )),
        };
    }

    /** @return list<int> */
    private function parseShape(mixed $value, string $name): array
    {
        if (!is_array($value) || !array_is_list($value)) {
            throw new SerializationException(sprintf('Tensor "%s" has no valid shape.', $name));
        }

        $dimensions = [];

        foreach ($value as $dimension) {
            if (!is_int($dimension) || $dimension < 0) {
                throw new SerializationException(sprintf(
                    'Tensor "%s" shape dimensions must be non-negative integers.',
                    $name,
                ));
            }

            $dimensions[] = $dimension;
        }

        return $dimensions;
    }

    /** @return array{int, int} */
    private function parseOffsets(mixed $value, string $name): array
    {
        if (
            !is_array($value)
            || count($value) !== 2
            || !array_is_list($value)
            || !is_int($value[0])
            || !is_int($value[1])
            || $value[0] < 0
            || $value[1] < $value[0]
        ) {
            throw new SerializationException(sprintf('Tensor "%s" has invalid data offsets.', $name));
        }

        return [$value[0], $value[1]];
    }

    /** @param list<int> $shape */
    private function expectedByteLength(array $shape, DType $dtype, string $name): int
    {
        $elements = 1;

        foreach ($shape as $dimension) {
            $elements = $this->checkedMultiply($elements, $dimension, sprintf('shape of tensor "%s"', $name));
        }

        $bytesPerElement = match ($dtype) {
            DType::Float32 => 4,
            DType::Float16, DType::BFloat16 => 2,
            DType::Int64 => 8,
            DType::Int8 => 1,
            DType::Int4 => throw new SerializationException('Int4 is not a supported Safetensors storage dtype.'),
        };

        return $this->checkedMultiply($elements, $bytesPerElement, sprintf('byte length of tensor "%s"', $name));
    }

    private function checkedAdd(int $left, int $right, string $context): int
    {
        if ($right > PHP_INT_MAX - $left) {
            throw new SerializationException(sprintf('Integer overflow while calculating %s.', $context));
        }

        return $left + $right;
    }

    private function checkedMultiply(int $left, int $right, string $context): int
    {
        if ($left !== 0 && $right > intdiv(PHP_INT_MAX, $left)) {
            throw new SerializationException(sprintf('Integer overflow while calculating %s.', $context));
        }

        return $left * $right;
    }
}
