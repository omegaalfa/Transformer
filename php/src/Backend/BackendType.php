<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Backend;

enum BackendType: string
{
    case PurePhp = 'pure-php';
    case Ffi = 'ffi';
    case NativeExtension = 'native-extension';
}
