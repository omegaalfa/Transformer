<?php

declare(strict_types=1);

use Omegaalfa\Transformer\Backend\BackendType;
use Omegaalfa\Transformer\Backend\Cuda\CudaBgePrecision;
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
$precisionName = getenv('TRANSFORMER_BGE_MIXED_PRECISION') ?: 'float16';
$precision = match (strtolower($precisionName)) {
    'float16', 'fp16' => CudaBgePrecision::Float16,
    'bfloat16', 'bf16' => CudaBgePrecision::BFloat16,
    default => throw new InvalidArgumentException('Precision must be float16/fp16 or bfloat16/bf16.'),
};
$model = (new CudaBgeEmbeddingModelLoader($runtime, $library, $precision))->load($checkpoint);
$graphSetting = getenv('TRANSFORMER_BGE_MIXED_GRAPH');
$graphEnabled = !is_string($graphSetting) || $graphSetting !== '0';
$model->library->setGraphEnabled($graphEnabled);
$short = 'hello world';
$long = array_fill(0, 8, implode(' ', array_fill(0, 8, 'The quick brown fox jumps over the lazy dog.')));

for ($index = 0; $index < 100; ++$index) {
    $model->encode($short);
}
$shortFirst = $model->encode($short);
$shortBefore = $model->library->memoryInfo();
for ($index = 0; $index < 1_000; ++$index) {
    $shortLast = $model->encode($short);
}
$shortAfter = $model->library->memoryInfo();

for ($index = 0; $index < 100; ++$index) {
    $model->encodeBatch($long);
}
$longFirst = $model->encodeBatch($long);
$longBefore = $model->library->memoryInfo();
for ($index = 0; $index < 100; ++$index) {
    $longLast = $model->encodeBatch($long);
}
$longAfter = $model->library->memoryInfo();

echo json_encode([
    'precision' => $model->library->precision->name,
    'graph_enabled' => $graphEnabled,
    'short' => ['forwards' => 1_000, 'deterministic' => $shortFirst === $shortLast,
        'free_before' => $shortBefore['free'], 'free_after' => $shortAfter['free'],
        'delta' => $shortAfter['free'] - $shortBefore['free']],
    'batch_long' => ['batch' => 8, 'forwards' => 100, 'bitwise_deterministic' => $longFirst === $longLast,
        'repeat_comparison' => compareRows($longFirst, $longLast),
        'free_before' => $longBefore['free'], 'free_after' => $longAfter['free'],
        'delta' => $longAfter['free'] - $longBefore['free']],
    'diagnostics' => $model->library->benchmarkDiagnostics(),
], JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR) . PHP_EOL;

/** @param list<list<float>> $left @param list<list<float>> $right @return array{max_abs: float, changed: int} */
function compareRows(array $left, array $right): array
{
    $max = 0.0;
    $changed = 0;
    foreach ($left as $row => $values) {
        foreach ($values as $column => $value) {
            $error = abs($value - $right[$row][$column]);
            $max = max($max, $error);
            $changed += $error > 0.0 ? 1 : 0;
        }
    }

    return ['max_abs' => $max, 'changed' => $changed];
}
