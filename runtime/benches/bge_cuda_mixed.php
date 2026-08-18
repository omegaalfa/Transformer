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
$samples = max(100, (int) (getenv('TRANSFORMER_BGE_MIXED_SAMPLES') ?: 100));
$warmups = max(100, (int) (getenv('TRANSFORMER_BGE_MIXED_WARMUPS') ?: 100));
$library = NativeLibrary::defaultPath($root);
$runtime = new Runtime(new FfiBackend(new NativeLibrary($library)), new RuntimeConfig(BackendType::Ffi));
$texts = [
    'short' => 'hello world',
    'medium' => implode(' ', array_fill(0, 3, 'The quick brown fox jumps over the lazy dog.')),
    'long' => implode(' ', array_fill(0, 8, 'The quick brown fox jumps over the lazy dog.')),
];
$batches = [1, 8, 32];
$models = $loads = $results = [];
foreach (CudaBgePrecision::cases() as $precision) {
    $loaded = (new CudaBgeEmbeddingModelLoader($runtime, $library, $precision))->benchmarkLoad($checkpoint);
    $models[$precision->name] = $loaded['model'];
    $loads[$precision->name] = $loaded['timings_us'];
}

foreach ($texts as $size => $text) {
    foreach ($batches as $batch) {
        if ($batch === 32 && $size === 'long') {
            continue;
        }
        $case = "B{$batch}_{$size}";
        $input = array_fill(0, $batch, $text);
        $tokens = $models['Float32']->tokenizer->encodeBatch($input);
        [, $sequence] = $tokens->shape->dimensions;
        foreach ($models as $name => $model) {
            for ($index = 0; $index < $warmups; ++$index) {
                $model->encodeBatch($input);
            }
            $times = [];
            $last = [];
            for ($index = 0; $index < $samples; ++$index) {
                $start = hrtime(true);
                $last = $model->encodeBatch($input);
                $times[] = (hrtime(true) - $start) / 1_000;
            }
            $device = $model->library->benchmarkForward(
                $tokens->inputIds,
                $tokens->attentionMask->values,
                $tokens->tokenTypeIds,
                $batch,
                $sequence,
            )['phases_us']['device'];
            $stats = statistics($times);
            $results[$case][$name] = [
                'batch' => $batch,
                'sequence' => $sequence,
                'latency_us' => $stats,
                'device_us' => $device,
                'sentences_per_second' => $batch * 1_000_000 / $stats['p50'],
                'tokens_per_second' => $batch * $sequence * 1_000_000 / $stats['p50'],
                'output' => $last,
                'diagnostics' => $model->library->benchmarkDiagnostics(),
            ];
        }
        $reference = flatten($results[$case]['Float32']['output']);
        foreach ($results[$case] as &$entry) {
            $entry['parity_fp32'] = compare($reference, flatten($entry['output']));
            unset($entry['output']);
        }
        unset($entry);
    }
}

$semanticTexts = [
    'A' => 'A man is eating food.',
    'B' => 'A person eats a meal.',
    'C' => 'Quantum mechanics describes subatomic particles.',
];
$semantic = [];
foreach ($models as $name => $model) {
    $vectors = $model->encodeBatch(array_values($semanticTexts));
    $ab = dot($vectors[0], $vectors[1]);
    $ac = dot($vectors[0], $vectors[2]);
    $semantic[$name] = ['similar_ab' => $ab, 'unrelated_ac' => $ac, 'margin' => $ab - $ac, 'ranking_passed' => $ab > $ac];
}

echo json_encode([
    'checkpoint' => $checkpoint,
    'warmups' => $warmups,
    'samples' => $samples,
    'cold_load_us' => $loads,
    'results' => $results,
    'semantic' => $semantic,
], JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR) . PHP_EOL;

/** @param list<float> $values @return array<string, float> */
function statistics(array $values): array
{
    sort($values, SORT_NUMERIC);
    $mean = array_sum($values) / count($values);
    $sum = 0.0;
    foreach ($values as $value) {
        $sum += ($value - $mean) ** 2;
    }

    return ['p50' => percentile($values, 0.50), 'p95' => percentile($values, 0.95),
        'p99' => percentile($values, 0.99), 'mean' => $mean, 'stddev' => sqrt($sum / count($values)),
        'min' => $values[0], 'max' => $values[array_key_last($values)]];
}

/** @param list<float> $values */
function percentile(array $values, float $fraction): float
{
    return $values[(int) ceil(count($values) * $fraction) - 1];
}

/** @param list<list<float>> $rows @return list<float> */
function flatten(array $rows): array
{
    return array_merge(...$rows);
}

/** @param list<float> $expected @param list<float> $actual @return array<string, float|bool> */
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

    $cosine = $dot / sqrt($left * $right);

    return ['max_abs' => $max, 'mean_abs' => $sum / count($expected), 'max_rel' => $relative,
        'cosine' => $cosine, 'passed' => $cosine >= 0.9999];
}

/** @param list<float> $left @param list<float> $right */
function dot(array $left, array $right): float
{
    $sum = 0.0;
    foreach ($left as $index => $value) {
        $sum += $value * $right[$index];
    }

    return $sum;
}
