<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Transformer\Config;

final readonly class TransformerConfig
{
    public function __construct(
        public int $hiddenSize,
        public int $attentionHeads,
        public int $intermediateSize,
        public int $layers,
        public float $normalizationEpsilon = 1.0e-5,
    ) {
    }
}
