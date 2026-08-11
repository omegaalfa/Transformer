<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tokenizer;

final readonly class TokenizationResult
{
    /**
     * @param list<int> $tokenIds
     * @param list<int> $attentionMask
     * @param list<int>|null $tokenTypeIds
     */
    public function __construct(
        public array $tokenIds,
        public array $attentionMask,
        public ?array $tokenTypeIds = null,
    ) {
    }
}
