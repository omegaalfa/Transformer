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

// Input, weights e residual permanecem residentes. Apenas os quatro resultados
// da inferência são temporários e são liberados ao fim de cada iteração.
$infer = static function () use ($input, $weights, $residual) {
    $projected = $input->matmul($weights);
    $added = $projected->add($residual);
    $normalized = $added->softmax();
    $output = $normalized->transpose();
    foreach ([$normalized, $added, $projected] as $temporary) {
        $temporary->destroy();
    }

    return $output;
};

$repetitions = max(1, (int) (getenv('TRANSFORMER_EXAMPLE_REPETITIONS') ?: 3));
$output = null;
for ($iteration = 0; $iteration < $repetitions; ++$iteration) {
    $output?->destroy();
    $output = $infer();
}

// Única materialização para um array PHP, no final do pipeline.
$values = $output->toFloat32();

echo "Pipeline: matmul -> add -> softmax -> transpose\n";
echo "inferências com residentes: {$repetitions}\n";
echo 'shape final: ', json_encode($output->shape()->dimensions, JSON_THROW_ON_ERROR), "\n";
echo 'resultado: ', json_encode(array_chunk($values, 2), JSON_THROW_ON_ERROR), "\n";

// Teardown explícito. Destruir os residentes ocorre somente depois da última
// inferência; o destrutor PHP continua sendo o fallback em caso de exceção.
foreach ([$output, $residual, $weights, $input] as $tensor) {
    $tensor->destroy();
}
