<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Embedding\Pooling;

use LogicException;
use Omegaalfa\Transformer\Tensor\Tensor;

final class ClsPooling implements PoolingInterface
{
    public function pool(Tensor $hiddenStates, ?Tensor $attentionMask = null): Tensor
    {
        throw new LogicException('CLS pooling is not implemented yet.');
    }
}
