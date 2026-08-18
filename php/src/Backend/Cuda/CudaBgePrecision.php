<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Backend\Cuda;

enum CudaBgePrecision: int
{
    case Float32 = 0;
    case Float16 = 1;
    case BFloat16 = 2;
}
