<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Backend\Contract;

use Omegaalfa\Transformer\Tensor\Tensor;
use Omegaalfa\Transformer\Transformer\AttentionMask;

interface BertBackendInterface
{
    public function exactGelu(Tensor $input): Tensor;

    public function bertSelfAttention(
        Tensor $input,
        Tensor $qWeight,
        Tensor $qBias,
        Tensor $kWeight,
        Tensor $kBias,
        Tensor $vWeight,
        Tensor $vBias,
        Tensor $outWeight,
        Tensor $outBias,
        int $heads,
        ?AttentionMask $mask = null,
    ): Tensor;
}
