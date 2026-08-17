<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Serialization\Safetensors;

interface SafetensorsReaderInterface
{
    public function metadata(string $path): WeightMap;

    public function tensor(string $path, string $name): SerializedTensor;
}
