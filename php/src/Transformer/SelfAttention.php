<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Transformer;

use LogicException;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Tensor\Tensor;
use Omegaalfa\Transformer\Transformer\Config\TransformerConfig;

final readonly class SelfAttention
{
    public function __construct(public TransformerConfig $config, public Runtime $runtime)
    {
    }

    public function forward(Tensor $input, ?AttentionMask $mask = null): Tensor
    {
        throw new LogicException('Self-attention is not implemented yet.');
    }
}
