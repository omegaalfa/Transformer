<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Serialization\Safetensors;

final readonly class WeightMap
{
    /** @param array<string, TensorMetadata> $tensors */
    public function __construct(public array $tensors)
    {
    }
}
