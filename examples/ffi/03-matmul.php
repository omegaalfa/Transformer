<?php

declare(strict_types=1);

use Omegaalfa\Transformer\Backend\Ffi\FfiBackend;
use Omegaalfa\Transformer\Backend\Ffi\NativeLibrary;

require dirname(__DIR__, 2) . '/vendor/autoload.php';

$projectRoot = dirname(__DIR__, 2);
$backend = new FfiBackend(new NativeLibrary(NativeLibrary::defaultPath($projectRoot)));
$a = [1, 2, 3, 4, 5, 6];
$b = [7, 8, 9, 10, 11, 12];
$output = $backend->matmulFloat32($a, $b, m: 2, k: 3, n: 2);

echo "A [2 x 3]:\n", json_encode(array_chunk($a, 3), JSON_THROW_ON_ERROR), "\n\n";
echo "B [3 x 2]:\n", json_encode(array_chunk($b, 2), JSON_THROW_ON_ERROR), "\n\n";
echo "A x B [2 x 2]:\n", json_encode(array_chunk($output, 2), JSON_THROW_ON_ERROR), "\n";
