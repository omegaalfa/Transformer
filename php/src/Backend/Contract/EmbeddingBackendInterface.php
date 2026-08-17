<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Backend\Contract;

use Omegaalfa\Transformer\Tensor\Shape;
use Omegaalfa\Transformer\Tensor\Tensor;

interface EmbeddingBackendInterface
{
    /** @param list<int> $tokenIds */
    public function embeddingTokenIds(array $tokenIds, Shape $shape, Tensor $weight): Tensor;
}
