<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Embedding;

use Omegaalfa\Transformer\Tensor\Tensor;

final readonly class BgeEmbeddingOutput
{
    public function __construct(public Tensor $pooled, public Tensor $embedding)
    {
    }
}
