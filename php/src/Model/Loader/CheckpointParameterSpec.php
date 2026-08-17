<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Model\Loader;

use InvalidArgumentException;

final readonly class CheckpointParameterSpec
{
    public function __construct(
        public string $checkpointName,
        public string $parameterName,
        public WeightMaterializationSpec $materialization,
    ) {
        if ($checkpointName === '' || $parameterName === '') {
            throw new InvalidArgumentException('Checkpoint and Parameter names must not be empty.');
        }
    }
}
