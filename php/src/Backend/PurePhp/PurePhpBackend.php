<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Backend\PurePhp;

use Omegaalfa\Transformer\Backend\AbstractBackend;
use Omegaalfa\Transformer\Backend\BackendType;

final class PurePhpBackend extends AbstractBackend
{
    public function type(): BackendType
    {
        return BackendType::PurePhp;
    }
}
