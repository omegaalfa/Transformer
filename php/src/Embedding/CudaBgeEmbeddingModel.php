<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Embedding;

use Omegaalfa\Transformer\Backend\Cuda\CudaBgeLibrary;
use Omegaalfa\Transformer\Tokenizer\BertTokenizer;

final readonly class CudaBgeEmbeddingModel implements EmbeddingModelInterface
{
    public function __construct(public CudaBgeLibrary $library, public BertTokenizer $tokenizer)
    {
    }

    /** @return list<float> */
    public function encode(string $text): array
    {
        return $this->encodeBatch([$text])[0];
    }

    /** @param list<string> $texts @return list<list<float>> */
    public function encodeBatch(array $texts): array
    {
        $tokens = $this->tokenizer->encodeBatch($texts);
        [$batch, $sequence] = $tokens->shape->dimensions;

        return $this->library->forward(
            $tokens->inputIds,
            $tokens->attentionMask->values,
            $tokens->tokenTypeIds,
            $batch,
            $sequence,
        );
    }
}
