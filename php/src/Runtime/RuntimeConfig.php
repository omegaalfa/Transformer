<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Runtime;

use Omegaalfa\Transformer\Backend\BackendType;

final readonly class RuntimeConfig
{
    public function __construct(public BackendType $backend = BackendType::PurePhp)
    {
    }
}
