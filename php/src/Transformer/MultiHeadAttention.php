<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Transformer;

use LogicException;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Tensor\Tensor;
use Omegaalfa\Transformer\Transformer\Config\TransformerConfig;

final readonly class MultiHeadAttention
{
    public function __construct(public TransformerConfig $config, public Runtime $runtime)
    {
    }

    public function forward(Tensor $input, ?AttentionMask $mask = null): Tensor
    {
        throw new LogicException('Multi-head attention is not implemented yet.');
    }
}
