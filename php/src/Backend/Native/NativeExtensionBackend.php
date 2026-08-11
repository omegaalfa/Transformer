<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Backend\Native;

use Omegaalfa\Transformer\Backend\AbstractBackend;
use Omegaalfa\Transformer\Backend\BackendType;

final class NativeExtensionBackend extends AbstractBackend
{
    public function type(): BackendType
    {
        return BackendType::NativeExtension;
    }
}
