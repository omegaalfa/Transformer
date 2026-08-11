<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Model\Loader;

use Omegaalfa\Transformer\Tensor\Tensor;

interface WeightLoaderInterface
{
    /** @return array<string, Tensor> */
    public function load(string $path): array;
}
