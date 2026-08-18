<?php

declare(strict_types=1);

use Omegaalfa\Transformer\Backend\BackendType;
use Omegaalfa\Transformer\Backend\Ffi\FfiBackend;
use Omegaalfa\Transformer\Backend\Ffi\NativeLibrary;
use Omegaalfa\Transformer\Model\Loader\CudaBgeEmbeddingModelLoader;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Runtime\RuntimeConfig;

require dirname(__DIR__, 2) . '/vendor/autoload.php';
$root = dirname(__DIR__, 2);
$checkpoint = getenv('TRANSFORMER_BGE_CUDA_CHECKPOINT') ?: '/tmp/transformer-model-r3/bge-small-en-v1.5';
$library = NativeLibrary::defaultPath($root);
$runtime = new Runtime(new FfiBackend(new NativeLibrary($library)), new RuntimeConfig(BackendType::Ffi));
$model = (new CudaBgeEmbeddingModelLoader($runtime, $library))->load($checkpoint);
$cases = [
    'short_1000' => ['hello world', 1000],
    'long_100' => [implode(' ', array_fill(0, 8, 'The quick brown fox jumps over lazy dogs.')), 100],
];
$report = [];
foreach ($cases as $name => [$text, $count]) {
    $first = $model->encode($text);
    $model->encode($text);
    $before = $model->library->memoryInfo();
    for ($index = 0; $index < $count; ++$index) {
        $last = $model->encode($text);
    }
    $after = $model->library->memoryInfo();
    $report[$name] = [
        'forwards' => $count,
        'deterministic' => $first === $last,
        'parameter_count' => $model->library->parameterCount(),
        'handle_identity' => $model->library->identity(),
        'free_bytes_before' => $before['free'],
        'free_bytes_after' => $after['free'],
        'free_bytes_delta' => $after['free'] - $before['free'],
    ];
}
echo json_encode($report, JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR) . PHP_EOL;
