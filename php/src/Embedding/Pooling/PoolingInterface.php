<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Embedding\Pooling;

use Omegaalfa\Transformer\Tensor\Tensor;

interface PoolingInterface
{
    public function pool(Tensor $hiddenStates, ?Tensor $attentionMask = null): Tensor;
}
