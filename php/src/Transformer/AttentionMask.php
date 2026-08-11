<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Transformer;

use Omegaalfa\Transformer\Tensor\Tensor;

final readonly class AttentionMask
{
    public function __construct(public Tensor $tensor)
    {
    }
}
