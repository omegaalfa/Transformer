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
$outputs = [];
foreach (CudaBgePrecision::cases() as $precision) {
    $start = hrtime(true);
    $model = (new CudaBgeEmbeddingModelLoader($runtime, $library, $precision))->load($checkpoint);
    $first = $model->encode('hello world');
    $model->encode('hello world');
    $output = $model->encode('hello world');
    $outputs[$precision->name] = ['load_us' => (hrtime(true) - $start) / 1_000, 'output' => $output,
        'deterministic' => $first === $output, 'diagnostics' => $model->library->benchmarkDiagnostics()];
    unset($model);
}
$reference = $outputs['Float32']['output'];
foreach ($outputs as &$entry) {
    $entry['comparison'] = compare($reference, $entry['output']);
    unset($entry['output']);
}
echo json_encode($outputs, JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR) . PHP_EOL;

/** @param list<float> $expected @param list<float> $actual @return array<string, float> */
function compare(array $expected, array $actual): array
{
    $max = $sum = $relative = $dot = $left = $right = 0.0;
    foreach ($expected as $index => $value) {
        $error = abs($actual[$index] - $value);
        $max = max($max, $error);
        $sum += $error;
        if (abs($value) >= 1.0e-3) {
            $relative = max($relative, $error / abs($value));
        }
        $dot += $value * $actual[$index];
        $left += $value * $value;
        $right += $actual[$index] * $actual[$index];
    }
    return ['max_abs' => $max, 'mean_abs' => $sum / count($expected), 'max_rel' => $relative,
        'cosine' => $dot / sqrt($left * $right), 'l2_norm' => sqrt($right)];
}
