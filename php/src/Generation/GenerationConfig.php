<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Generation;

final readonly class GenerationConfig
{
    public function __construct(
        public int $maximumNewTokens,
        public float $temperature = 1.0,
        public ?int $topK = null,
        public ?float $topP = null,
    ) {
    }
}
