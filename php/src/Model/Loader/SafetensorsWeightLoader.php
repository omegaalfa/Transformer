<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Model\Loader;

use Omegaalfa\Transformer\Exception\SerializationException;
use Omegaalfa\Transformer\NN\Parameter;
use Omegaalfa\Transformer\Serialization\Safetensors\SafetensorsReader;
use Omegaalfa\Transformer\Serialization\Safetensors\SafetensorsReaderInterface;
use Omegaalfa\Transformer\Serialization\Safetensors\SerializedTensor;

final readonly class SafetensorsWeightLoader implements WeightLoaderInterface
{
    public function __construct(
        private SafetensorsReaderInterface $reader,
        private WeightMaterializer $materializer,
        private WeightManifest $manifest,
    ) {
    }

    public function load(string $path): array
    {
        if ($this->reader instanceof SafetensorsReader) {
            $session = $this->reader->open($path);
            $weightMap = $session->weightMap;
        } else {
            $session = null;
            $weightMap = $this->reader->metadata($path);
        }
        $expectedNames = [];

        foreach ($this->manifest->parameters as $parameter) {
            $expectedNames[$parameter->checkpointName] = true;
        }
        foreach ($this->manifest->ignoredCheckpointTensors as $name) {
            $expectedNames[$name] = true;
        }

        $actualNames = array_fill_keys(array_keys($weightMap->tensors), true);
        $missing = array_keys(array_diff_key($expectedNames, $actualNames));
        $unexpected = array_keys(array_diff_key($actualNames, $expectedNames));

        if ($missing !== []) {
            throw new SerializationException(sprintf(
                'Safetensors checkpoint is missing required tensors: %s.',
                implode(', ', $missing),
            ));
        }
        if ($unexpected !== []) {
            throw new SerializationException(sprintf(
                'Safetensors checkpoint contains unexpected tensors: %s.',
                implode(', ', $unexpected),
            ));
        }

        foreach ($this->manifest->parameters as $parameter) {
            $metadata = $weightMap->tensors[$parameter->checkpointName];

            if ($metadata->shape->dimensions !== $parameter->materialization->checkpointShape->dimensions) {
                throw new SerializationException(sprintf(
                    'Tensor "%s" shape does not match the weight manifest.',
                    $parameter->checkpointName,
                ));
            }
        }

        $loaded = [];

        foreach ($this->manifest->parameters as $parameter) {
            $serialized = $session === null
                ? $this->reader->tensor($path, $parameter->checkpointName)
                : new SerializedTensor($weightMap->tensors[$parameter->checkpointName], $session->read($parameter->checkpointName));
            $tensor = $this->materializer->materialize($serialized, $parameter->materialization);
            $loaded[$parameter->parameterName] = new Parameter(
                name: $parameter->parameterName,
                tensor: $tensor,
                trainable: false,
            );
        }

        return $loaded;
    }
}
