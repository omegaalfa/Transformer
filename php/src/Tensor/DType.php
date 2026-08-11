<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tensor;

enum DType: string
{
    case Float32 = 'float32';
    case Float16 = 'float16';
    case BFloat16 = 'bfloat16';
    case Int8 = 'int8';
    case Int4 = 'int4';
}
