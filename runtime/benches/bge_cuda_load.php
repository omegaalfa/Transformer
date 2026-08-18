<?php

declare(strict_types=1);

use Omegaalfa\Transformer\Backend\BackendType;
use Omegaalfa\Transformer\Backend\Cuda\CudaBgePrecision;
use Omegaalfa\Transformer\Backend\Ffi\FfiBackend;
use Omegaalfa\Transformer\Backend\Ffi\NativeLibrary;
use Omegaalfa\Transformer\Model\Loader\CudaBgeEmbeddingModelLoader;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Runtime\RuntimeConfig;

require dirname(__DIR__, 2).'/vendor/autoload.php';
$root = dirname(__DIR__, 2);
$checkpoint = getenv('TRANSFORMER_BGE_CUDA_CHECKPOINT') ?: '/tmp/transformer-model-r3/bge-small-en-v1.5';
$samples = max(5, (int) (getenv('TRANSFORMER_BGE_LOAD_SAMPLES') ?: 5));
$precision = match (strtolower(getenv('TRANSFORMER_BGE_LOAD_PRECISION') ?: 'fp32')) {
    'fp32', 'float32' => CudaBgePrecision::Float32,
    'fp16', 'float16' => CudaBgePrecision::Float16,
    'bf16', 'bfloat16' => CudaBgePrecision::BFloat16,
    default => throw new InvalidArgumentException('Unsupported load benchmark precision.'),
};
$library = NativeLibrary::defaultPath($root);
$runtime = new Runtime(new FfiBackend(new NativeLibrary($library)), new RuntimeConfig(BackendType::Ffi));
$loads = [];
for ($sample = 0; $sample < $samples; ++$sample) {
    $start = hrtime(true);
    $loaded = (new CudaBgeEmbeddingModelLoader($runtime, $library, $precision))->benchmarkLoad($checkpoint);
    $total = (hrtime(true) - $start) / 1_000;
    $model = $loaded['model'];
    $phases = $loaded['timings_us'];
    $output = $model->encode('hello world');
    $vramLoaded = $model->library->memoryInfo();
    $diagnostics = $model->library->benchmarkDiagnostics();
    $model->library->destroy();
    $vramDestroyed = $model->library->memoryInfo();
    unset($model, $loaded);
    gc_collect_cycles();
    $loads[] = ['total_us' => $total, 'phases_us' => $phases,
        'output_finite' => count(array_filter($output, is_finite(...))) === 384,
        'parameter_bytes' => $diagnostics['parameter_bytes'],
        'vram_loaded_free' => $vramLoaded['free'], 'vram_destroyed_free' => $vramDestroyed['free']];
}
$times = array_column($loads, 'total_us');
sort($times, SORT_NUMERIC);
echo json_encode(['precision' => $precision->name, 'samples' => $samples,
    'first_coldish_us' => $loads[0]['total_us'], 'warm_page_cache_us' => [
        'p50' => percentile($times, 0.50), 'p95' => percentile($times, 0.95),
        'min' => $times[0], 'max' => $times[array_key_last($times)],
    ], 'php_peak_bytes' => memory_get_peak_usage(true), 'rss_high_water_bytes' => rssHighWaterBytes(),
    'loads' => $loads], JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR).PHP_EOL;

/** @param list<float> $values */
function percentile(array $values, float $fraction): float
{
    return $values[(int) ceil(count($values) * $fraction) - 1];
}

function rssHighWaterBytes(): ?int
{
    $status = @file_get_contents('/proc/self/status');
    if (!is_string($status) || preg_match('/^VmHWM:\s+(\d+) kB$/m', $status, $matches) !== 1) {
        return null;
    }

    return (int) $matches[1] * 1024;
}
