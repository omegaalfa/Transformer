<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Serialization\Safetensors;

final readonly class SerializedTensor
{
    public function __construct(
        public TensorMetadata $metadata,
        public string $bytes,
    ) {
    }
}
