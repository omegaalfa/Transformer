<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Serialization\Safetensors;

use Omegaalfa\Transformer\Exception\SerializationException;

final class SafetensorsReadSession
{
    /** @param resource $handle */
    public function __construct(private $handle, public readonly WeightMap $weightMap)
    {
    }

    public function __destruct()
    {
        if (is_resource($this->handle)) {
            fclose($this->handle);
        }
    }

    public function read(string $name): string
    {
        $metadata = $this->weightMap->tensors[$name] ?? null;
        if ($metadata === null) {
            throw new SerializationException(sprintf('Safetensors tensor "%s" does not exist.', $name));
        }
        if (fseek($this->handle, $metadata->offset, SEEK_SET) !== 0) {
            throw new SerializationException(sprintf('Unable to seek to Safetensors tensor "%s".', $name));
        }

        $contents = '';
        while (strlen($contents) < $metadata->length) {
            $remaining = $metadata->length - strlen($contents);
            if ($remaining < 1) {
                break;
            }
            $chunk = fread($this->handle, $remaining);
            if ($chunk === false || $chunk === '') {
                throw new SerializationException(sprintf('Truncated Safetensors tensor "%s" payload.', $name));
            }
            $contents .= $chunk;
        }

        return $contents;
    }
}
