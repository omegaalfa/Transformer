<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Embedding;

interface EmbeddingModelInterface
{
    /** @return list<float> */
    public function encode(string $text): array;

    /** @param list<string> $texts
     *  @return list<list<float>>
     */
    public function encodeBatch(array $texts): array;
}
