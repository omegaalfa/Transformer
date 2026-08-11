<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Embedding;

use Omegaalfa\Transformer\Tensor\Tensor;

interface EmbeddingModelInterface
{
    public function encode(string $text): Tensor;
}
