<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tensor;

enum Device: string
{
    case CPU = 'cpu';
    case CUDA = 'cuda';
}
