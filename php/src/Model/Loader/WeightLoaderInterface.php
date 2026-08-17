<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Model\Loader;

use Omegaalfa\Transformer\NN\Parameter;

interface WeightLoaderInterface
{
    /** @return array<string, Parameter> */
    public function load(string $path): array;
}
