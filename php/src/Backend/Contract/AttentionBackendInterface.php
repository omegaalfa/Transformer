<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Backend\Contract;

use Omegaalfa\Transformer\Tensor\Tensor;
use Omegaalfa\Transformer\Transformer\AttentionMask;

interface AttentionBackendInterface
{
    public function multiHeadAttention(
        Tensor $input,
        Tensor $qWeight,
        Tensor $kWeight,
        Tensor $vWeight,
        Tensor $outWeight,
        int $heads,
        ?AttentionMask $mask = null,
    ): Tensor;
}
