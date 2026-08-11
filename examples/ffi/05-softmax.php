<?php

declare(strict_types=1);

use Omegaalfa\Transformer\Backend\Ffi\FfiBackend;
use Omegaalfa\Transformer\Backend\Ffi\NativeLibrary;

require dirname(__DIR__, 2) . '/vendor/autoload.php';

$projectRoot = dirname(__DIR__, 2);
$backend = new FfiBackend(new NativeLibrary(NativeLibrary::defaultPath($projectRoot)));
$input = [1, 2, 3];
$output = $backend->softmaxFloat32($input);

echo 'input:   ', json_encode($input, JSON_THROW_ON_ERROR), "\n";
echo 'softmax: ', json_encode($output, JSON_THROW_ON_ERROR), "\n";
echo 'sum:     ', array_sum($output), "\n";
