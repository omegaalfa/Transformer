<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tokenizer;

final readonly class Vocabulary
{
    /** @param array<string, int> $tokens */
    public function __construct(public array $tokens)
    {
    }
}
