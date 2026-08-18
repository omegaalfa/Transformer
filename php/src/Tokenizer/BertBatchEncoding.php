<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tokenizer;

use Omegaalfa\Transformer\Tensor\Shape;
use Omegaalfa\Transformer\Transformer\AttentionMask;

final readonly class BertBatchEncoding
{
    /**
     * @param list<int> $inputIds
     * @param list<int> $tokenTypeIds
     */
    public function __construct(
        public array $inputIds,
        public AttentionMask $attentionMask,
        public array $tokenTypeIds,
        public Shape $shape,
    ) {
    }
}
