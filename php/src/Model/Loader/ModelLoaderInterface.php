<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Model\Loader;

use Omegaalfa\Transformer\Model\ModelConfig;
use Omegaalfa\Transformer\Model\ModelInterface;

interface ModelLoaderInterface
{
    public function load(string $path, ModelConfig $config): ModelInterface;
}
