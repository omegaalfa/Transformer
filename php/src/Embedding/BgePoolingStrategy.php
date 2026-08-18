<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Embedding;

enum BgePoolingStrategy: string
{
    case Cls = 'cls';
    case Mean = 'mean';
}
