<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Transformer;

use LogicException;
use Omegaalfa\Transformer\Tensor\Tensor;

final readonly class TransformerBlock
{
    public function __construct(
        public MultiHeadAttention $attention,
        public FeedForward $feedForward,
    ) {
    }

    public function forward(Tensor $input, ?AttentionMask $mask = null): Tensor
    {
        throw new LogicException('Transformer block is not implemented yet.');
    }
}
