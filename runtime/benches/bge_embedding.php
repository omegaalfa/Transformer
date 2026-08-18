<?php

declare(strict_types=1);

use Omegaalfa\Transformer\Backend\BackendType;
use Omegaalfa\Transformer\Backend\Ffi\FfiBackend;
use Omegaalfa\Transformer\Backend\Ffi\NativeLibrary;
use Omegaalfa\Transformer\Model\Loader\BgeEmbeddingModelLoader;
use Omegaalfa\Transformer\NN\Module;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Runtime\RuntimeConfig;

require dirname(__DIR__, 2) . '/vendor/autoload.php';

$root = dirname(__DIR__, 2);
$checkpoint = getenv('TRANSFORMER_BGE_CHECKPOINT') ?: '/tmp/transformer-model-r3/bge-small-en-v1.5';
$samples = max(3, (int) (getenv('TRANSFORMER_BGE_SAMPLES') ?: 15));
$recreateSamples = max(1, (int) (getenv('TRANSFORMER_BGE_RECREATE_SAMPLES') ?: 3));
$soakIterations = max(3, (int) (getenv('TRANSFORMER_BGE_SOAK_ITERATIONS') ?: 25));

$runtime = new Runtime(
    new FfiBackend(new NativeLibrary(NativeLibrary::defaultPath($root))),
    new RuntimeConfig(BackendType::Ffi),
);
$loader = new BgeEmbeddingModelLoader($runtime);
$rssBefore = rssBytes();
$loadStart = hrtime(true);
$model = $loader->load($checkpoint);
$coldLoadUs = elapsedUs($loadStart);
$rssAfterLoad = rssBytes();
$storageIds = parameterStorageIds($model->model);
$parity = parityMetrics(
    $model->encode('hello world'),
    dirname($checkpoint) . '/reference/hello_world/cls_normalized.f32',
);

$texts = [
    'short' => 'How do I request annual leave?',
    'medium' => implode(' ', array_fill(0, 2, 'The employee handbook explains the approval process and expected response time.')),
    'long' => implode(' ', array_fill(0, 4, 'This representative document sentence contains enough ordinary English words to exercise a longer BERT sequence.')),
];
$tokenCounts = [];
foreach ($texts as $name => $text) {
    $tokenCounts[$name] = count($model->tokenizer->encode($text)->tokenIds);
    $model->encode($text);
}
$rssAfterWarmup = rssBytes();

$singleStats = statistics(measure($samples, static fn () => $model->encode($texts['short'])));
$batchResults = [];
foreach ($texts as $length => $text) {
    foreach ([1, 2, 8, 16, 32] as $batchSize) {
        $batch = array_fill(0, $batchSize, $text);
        $stats = statistics(measure($samples, static fn () => $model->encodeBatch($batch)));
        $batchResults[$length][(string) $batchSize] = [
            'tokens_per_sentence' => $tokenCounts[$length],
            'latency_us' => $stats,
            'sentences_per_second' => $batchSize / ($stats['mean'] / 1_000_000),
        ];
    }
}

$resident = statistics(measure($recreateSamples, static fn () => $model->encode($texts['short'])));
$recreate = statistics(measure($recreateSamples, static function () use ($loader, $checkpoint, $texts): void {
    $temporary = $loader->load($checkpoint);
    $temporary->encode($texts['short']);
    unset($temporary);
    gc_collect_cycles();
}));

$rssSoak = [];
for ($i = 0; $i < $soakIterations; ++$i) {
    $model->encode($texts['short']);
    $rssSoak[] = rssBytes();
}
$rssAfterBenchmark = rssBytes();
$sameStorages = $storageIds === parameterStorageIds($model->model);

echo json_encode([
    'checkpoint' => $checkpoint,
    'samples' => $samples,
    'cold_load' => [
        'latency_us' => $coldLoadUs,
        'parameter_storages' => count($storageIds),
        'rss_before' => $rssBefore,
        'rss_after' => $rssAfterLoad,
        'php_peak_bytes' => memory_get_peak_usage(true),
    ],
    'tokens' => $tokenCounts,
    'parity' => $parity,
    'warm_single_short_us' => $singleStats,
    'batch' => $batchResults,
    'residence' => [
        'resident_us' => $resident,
        'recreate_us' => $recreate,
        'mean_speedup' => $recreate['mean'] / $resident['mean'],
        'parameter_storages_unchanged' => $sameStorages,
    ],
    'memory' => [
        'rss_after_load' => $rssAfterLoad,
        'rss_after_warmup' => $rssAfterWarmup,
        'rss_after_benchmark' => $rssAfterBenchmark,
        'soak_first' => $rssSoak[0],
        'soak_last' => $rssSoak[array_key_last($rssSoak)],
        'soak_min' => min($rssSoak),
        'soak_max' => max($rssSoak),
    ],
], JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR) . PHP_EOL;

/** @return list<float> */
function measure(int $samples, Closure $operation): array
{
    $times = [];
    for ($i = 0; $i < $samples; ++$i) {
        $start = hrtime(true);
        $operation();
        $times[] = elapsedUs($start);
    }

    return $times;
}

/** @param list<float> $values
 *  @return array{p50: float, p95: float, p99: float, mean: float, min: float, max: float}
 */
function statistics(array $values): array
{
    sort($values, SORT_NUMERIC);
    $count = count($values);
    $percentile = static fn (float $p): float => $values[(int) ceil($p * $count) - 1];

    return [
        'p50' => $percentile(0.50),
        'p95' => $percentile(0.95),
        'p99' => $percentile(0.99),
        'mean' => array_sum($values) / $count,
        'min' => $values[0],
        'max' => $values[$count - 1],
    ];
}

function elapsedUs(int $start): float
{
    return (hrtime(true) - $start) / 1_000;
}

function rssBytes(): int
{
    $status = file_get_contents('/proc/self/status');
    if ($status !== false && preg_match('/^VmRSS:\s+(\d+)\s+kB$/m', $status, $match) === 1) {
        return (int) $match[1] * 1024;
    }

    return memory_get_usage(true);
}

/** @return list<int> */
function parameterStorageIds(Module $module): array
{
    $ids = [];
    foreach ($module->parameters() as $parameter) {
        $ids[] = spl_object_id($parameter->tensor->storage());
    }
    foreach ($module->modules() as $child) {
        array_push($ids, ...parameterStorageIds($child));
    }

    return $ids;
}

/** @param list<float> $actual
 *  @return array{checked: bool, passed: bool, max_abs: float, max_rel: float}
 */
function parityMetrics(array $actual, string $referencePath): array
{
    if (!is_file($referencePath)) {
        return ['checked' => false, 'passed' => false, 'max_abs' => 0.0, 'max_rel' => 0.0];
    }
    $decoded = unpack('g*', (string) file_get_contents($referencePath));
    if ($decoded === false) {
        throw new RuntimeException('Cannot decode Float32 parity reference.');
    }
    $expected = array_values($decoded);
    $passed = count($actual) === count($expected);
    $maxAbs = $maxRel = 0.0;
    foreach ($expected as $index => $reference) {
        $error = abs($actual[$index] - $reference);
        $maxAbs = max($maxAbs, $error);
        if (abs($reference) >= 1.0e-3) {
            $maxRel = max($maxRel, $error / abs($reference));
        }
        $passed = $passed && $error <= 2.0e-5 + 2.0e-5 * abs($reference);
    }

    return ['checked' => true, 'passed' => $passed, 'max_abs' => $maxAbs, 'max_rel' => $maxRel];
}
