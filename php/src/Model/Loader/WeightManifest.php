<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Model\Loader;

use InvalidArgumentException;

final readonly class WeightManifest
{
    /** @var list<CheckpointParameterSpec> */
    public array $parameters;

    /** @var list<string> */
    public array $ignoredCheckpointTensors;

    /**
     * @param list<CheckpointParameterSpec> $parameters
     * @param list<string>                  $ignoredCheckpointTensors
     */
    public function __construct(array $parameters, array $ignoredCheckpointTensors = [])
    {
        $checkpointNames = [];
        $parameterNames = [];

        foreach ($parameters as $parameter) {
            if (isset($checkpointNames[$parameter->checkpointName])) {
                throw new InvalidArgumentException(sprintf(
                    'Duplicate checkpoint tensor name "%s" in weight manifest.',
                    $parameter->checkpointName,
                ));
            }
            if (isset($parameterNames[$parameter->parameterName])) {
                throw new InvalidArgumentException(sprintf(
                    'Duplicate Parameter name "%s" in weight manifest.',
                    $parameter->parameterName,
                ));
            }

            $checkpointNames[$parameter->checkpointName] = true;
            $parameterNames[$parameter->parameterName] = true;
        }

        foreach ($ignoredCheckpointTensors as $name) {
            if ($name === '' || isset($checkpointNames[$name])) {
                throw new InvalidArgumentException('Ignored checkpoint tensor names must be non-empty and disjoint from Parameters.');
            }
            $checkpointNames[$name] = true;
        }

        $this->parameters = $parameters;
        $this->ignoredCheckpointTensors = $ignoredCheckpointTensors;
    }
}
