<?php

declare(strict_types=1);

use Omegaalfa\Transformer\Backend\Ffi\FfiBackend;
use Omegaalfa\Transformer\Backend\Ffi\NativeLibrary;

require dirname(__DIR__, 2) . '/vendor/autoload.php';

$projectRoot = dirname(__DIR__, 2);
$backend = new FfiBackend(new NativeLibrary(NativeLibrary::defaultPath($projectRoot)));
$a = [1, 2, 3];
$b = [10, 20, 30];

echo "A:\n", json_encode($a, JSON_THROW_ON_ERROR), "\n\n";
echo "B:\n", json_encode($b, JSON_THROW_ON_ERROR), "\n\n";
echo "A + B:\n", json_encode($backend->addFloat32($a, $b), JSON_THROW_ON_ERROR), "\n";
