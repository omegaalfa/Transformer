<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tokenizer;

interface TokenizerInterface
{
    public function encode(string $text): TokenizationResult;

    /** @param list<int> $tokenIds */
    public function decode(array $tokenIds): string;
}
