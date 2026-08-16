<?php

declare(strict_types=1);

use Omegaalfa\Transformer\Backend\Ffi\FfiBackend;
use Omegaalfa\Transformer\Backend\Ffi\NativeLibrary;
use Omegaalfa\Transformer\Tensor\Shape;

require dirname(__DIR__, 2) . '/vendor/autoload.php';

$projectRoot = dirname(__DIR__, 2);
$backend = new FfiBackend(new NativeLibrary(NativeLibrary::defaultPath($projectRoot)));
$vector = [1, 2, 3];
$legacyOutput = $backend->softmaxFloat32($vector);

echo "API de array (rank 1):\n";
echo 'input:   ', json_encode($vector, JSON_THROW_ON_ERROR), "\n";
echo 'softmax: ', json_encode($legacyOutput, JSON_THROW_ON_ERROR), "\n";
echo 'sum:     ', array_sum($legacyOutput), "\n\n";

$matrix = $backend->tensorFromFloat32(
    [1, 2, 3, 4, 5, 6],
    new Shape([2, 3]),
);
$softmax = $matrix->softmax();
$rows = array_chunk($softmax->toFloat32(), 3);

echo "Tensor residente [2 x 3], softmax no último eixo:\n";
echo json_encode($rows, JSON_THROW_ON_ERROR | JSON_PRETTY_PRINT), "\n";
echo 'somas: ', json_encode(array_map(array_sum(...), $rows), JSON_THROW_ON_ERROR), "\n";

$softmax->destroy();
$matrix->destroy();
