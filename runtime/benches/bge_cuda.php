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
$samples = max(100, (int) (getenv('TRANSFORMER_BGE_CUDA_SAMPLES') ?: 250));
$warmups = max(100, (int) (getenv('TRANSFORMER_BGE_CUDA_WARMUPS') ?: 100));
$text = 'hello world';
$longText = implode(' ', array_fill(0, 8, 'The quick brown fox jumps over the lazy dog.'));
$library = NativeLibrary::defaultPath($root);
$runtime = new Runtime(new FfiBackend(new NativeLibrary($library)), new RuntimeConfig(BackendType::Ffi));
$loadStart = hrtime(true);
$loaded = (new CudaBgeEmbeddingModelLoader($runtime, $library))->benchmarkLoad($checkpoint);
$model = $loaded['model'];
$loadUs = elapsedUs($loadStart);
$tokens = $model->tokenizer->encodeBatch([$text]);
[$batch, $sequence] = $tokens->shape->dimensions;

$lifecycle = [];
for ($index = 0; $index < 3; ++$index) {
    $start = hrtime(true);
    $model->encode($text);
    $diagnostics = $model->library->benchmarkDiagnostics();
    $diagnostics['encode_us'] = elapsedUs($start);
    $lifecycle[] = $diagnostics;
}

$ab = [];
foreach ([false, true] as $graph) {
    $model->library->setGraphEnabled($graph);
    for ($index = 0; $index < $warmups; ++$index) {
        $model->encode($text);
    }
    $times = $cudaTimes = [];
    for ($index = 0; $index < $samples; ++$index) {
        $start = hrtime(true);
        $model->encode($text);
        $times[] = elapsedUs($start);
        if ($index < 25) {
            $detail = $model->library->benchmarkForward($tokens->inputIds, $tokens->attentionMask->values, $tokens->tokenTypeIds, $batch, $sequence);
            $cudaTimes[] = $detail['phases_us']['device'];
        }
    }
    $ab[$graph ? 'graph_on' : 'graph_off'] = [
        'encode_us' => statistics($times),
        'cuda_device_us' => statistics($cudaTimes),
        'diagnostics' => $model->library->benchmarkDiagnostics(),
    ];
}

$tokenizerTimes = $details = [];
for ($index = 0; $index < 25; ++$index) {
    $start = hrtime(true);
    $model->tokenizer->encodeBatch([$text]);
    $tokenizerTimes[] = elapsedUs($start);
    $details[] = $model->library->benchmarkForward($tokens->inputIds, $tokens->attentionMask->values, $tokens->tokenTypeIds, $batch, $sequence)['phases_us'];
}
$decomposition = ['tokenizer' => percentile($tokenizerTimes, 0.50)];
foreach (array_keys($details[0]) as $phase) {
    $decomposition[$phase] = percentile(array_column($details, $phase), 0.50);
}
$decomposition['ffi_non_cuda'] = max(0.0, $decomposition['ffi_total'] - $decomposition['h2d'] - $decomposition['device'] - $decomposition['d2h']);
$decomposition['explained_total'] = $decomposition['tokenizer'] + $decomposition['preparation'] + $decomposition['ffi_total'] + $decomposition['materialization'];

for ($index = 0; $index < $warmups; ++$index) {
    $model->encode($text);
}
$constantBefore = $model->library->memoryInfo();
$first = $model->encode($text);
for ($index = 0; $index < 1000; ++$index) {
    $last = $model->encode($text);
}
$constantAfter = $model->library->memoryInfo();
$smallBeforeGrowth = $constantAfter;
$model->encode($longText);
$model->encode($longText);
for ($index = 0; $index < $warmups; ++$index) {
    $model->encode($longText);
}
$grownBefore = $model->library->memoryInfo();
for ($index = 0; $index < 100; ++$index) {
    $model->encode($longText);
}
$grownAfter = $model->library->memoryInfo();
$actual = $model->encode($text);

echo json_encode([
    'baseline' => 'PUBLIC BGE CUDA WARM ENCODE', 'checkpoint' => $checkpoint,
    'text' => $text, 'batch' => $batch, 'sequence' => $sequence,
    'optimized_default' => true, 'warmups' => $warmups, 'samples' => $samples,
    'load_total_us' => $loadUs, 'cold_load_us' => $loaded['timings_us'],
    'graph_lifecycle' => $lifecycle, 'graph_ab' => $ab,
    'decomposition_p50_us' => $decomposition,
    'constant_shape_soak' => ['forwards' => 1000, 'deterministic' => $first === $last,
        'free_before' => $constantBefore['free'], 'free_after' => $constantAfter['free'],
        'free_delta' => $constantAfter['free'] - $constantBefore['free']],
    'workspace_growth' => ['small_free' => $smallBeforeGrowth['free'],
        'large_stabilized_free' => $grownBefore['free'],
        'expected_growth_bytes' => $smallBeforeGrowth['free'] - $grownBefore['free'],
        'large_repeated_free' => $grownAfter['free'],
        'post_growth_delta' => $grownAfter['free'] - $grownBefore['free']],
    'parity' => parity($actual, dirname($checkpoint) . '/reference/hello_world/cls_normalized.f32'),
], JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR) . PHP_EOL;

function elapsedUs(int $start): float
{
    return (hrtime(true) - $start) / 1_000;
}

/** @param list<float> $values @return array<string, float> */
function statistics(array $values): array
{
    sort($values, SORT_NUMERIC);
    $mean = array_sum($values) / count($values);
    $squared = 0.0;
    foreach ($values as $value) {
        $squared += ($value - $mean) ** 2;
    }
    return ['p50' => percentile($values, 0.50), 'p95' => percentile($values, 0.95),
        'p99' => percentile($values, 0.99), 'mean' => $mean,
        'stddev' => sqrt($squared / count($values)), 'min' => $values[0],
        'max' => $values[array_key_last($values)]];
}

/** @param list<float> $values */
function percentile(array $values, float $fraction): float
{
    sort($values, SORT_NUMERIC);
    return $values[(int) ceil(count($values) * $fraction) - 1];
}

/** @param list<float> $actual @return array<string, float|bool> */
function parity(array $actual, string $path): array
{
    $expected = unpack('g*', (string) file_get_contents($path));
    if ($expected === false) {
        throw new RuntimeException('Cannot decode parity reference.');
    }
    $passed = true;
    $maxAbs = $maxRel = $sumAbs = $dot = $actualNorm = $expectedNorm = 0.0;
    foreach (array_values($expected) as $index => $value) {
        $error = abs($actual[$index] - $value);
        $maxAbs = max($maxAbs, $error);
        $sumAbs += $error;
        $dot += $actual[$index] * $value;
        $actualNorm += $actual[$index] ** 2;
        $expectedNorm += $value ** 2;
        if (abs($value) >= 1.0e-3) {
            $maxRel = max($maxRel, $error / abs($value));
        }
        $passed = $passed && $error <= 2.0e-5 + 2.0e-5 * abs($value);
    }
    return ['passed' => $passed, 'max_abs' => $maxAbs, 'mean_abs' => $sumAbs / count($actual),
        'max_rel' => $maxRel, 'cosine' => $dot / sqrt($actualNorm * $expectedNorm)];
}
