<?php

declare(strict_types=1);

use Omegaalfa\Transformer\Backend\Ffi\FfiBackend;
use Omegaalfa\Transformer\Backend\Ffi\NativeLibrary;
use Omegaalfa\Transformer\Tensor\Shape;

require dirname(__DIR__, 2) . '/vendor/autoload.php';

$projectRoot = dirname(__DIR__, 2);
$backend = new FfiBackend(new NativeLibrary(NativeLibrary::defaultPath($projectRoot)));

$input = $backend->tensorFromFloat32([1, 2, 3, 4, 5, 6], new Shape([2, 3]));
$weights = $backend->tensorFromFloat32([1, 0, 0, 1, 1, 1], new Shape([3, 2]));
$residual = $backend->tensorFromFloat32([0.5, 0.5, 0.5, 0.5], new Shape([2, 2]));

// Todos os resultados intermediários continuam no runtime Rust.
$projected = $input->matmul($weights);
$added = $projected->add($residual);
$normalized = $added->softmax();
$output = $normalized->transpose();

// Única materialização para um array PHP, no final do pipeline.
$values = $output->toFloat32();

echo "Pipeline: matmul -> add -> softmax -> transpose\n";
echo 'shape final: ', json_encode($output->shape()->dimensions, JSON_THROW_ON_ERROR), "\n";
echo 'resultado: ', json_encode(array_chunk($values, 2), JSON_THROW_ON_ERROR), "\n";

// Liberação antecipada é opcional, mas útil para Tensors grandes.
foreach ([$output, $normalized, $added, $projected, $residual, $weights, $input] as $tensor) {
    $tensor->destroy();
}
