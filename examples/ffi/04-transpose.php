<?php

declare(strict_types=1);

use Omegaalfa\Transformer\Backend\Ffi\FfiBackend;
use Omegaalfa\Transformer\Backend\Ffi\NativeLibrary;

require dirname(__DIR__, 2) . '/vendor/autoload.php';

$projectRoot = dirname(__DIR__, 2);
$backend = new FfiBackend(new NativeLibrary(NativeLibrary::defaultPath($projectRoot)));
$input = [1, 2, 3, 4, 5, 6];
$output = $backend->transposeFloat32($input, rows: 2, columns: 3);

echo "A [2 x 3]:\n", json_encode(array_chunk($input, 3), JSON_THROW_ON_ERROR), "\n\n";
echo "transpose(A) [3 x 2]:\n", json_encode(array_chunk($output, 2), JSON_THROW_ON_ERROR), "\n";
