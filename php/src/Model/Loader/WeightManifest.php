<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Model\Loader;

use InvalidArgumentException;

final readonly class WeightManifest
{
    /** @var list<CheckpointParameterSpec> */
    public array $parameters;

    /** @param list<CheckpointParameterSpec> $parameters */
    public function __construct(array $parameters)
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

        $this->parameters = $parameters;
    }
}
