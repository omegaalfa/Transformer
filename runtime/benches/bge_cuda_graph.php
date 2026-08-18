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
$samples = max(5, (int) (getenv('TRANSFORMER_BGE_CUDA_GRAPH_SAMPLES') ?: 25));
$library = NativeLibrary::defaultPath($root);
$runtime = new Runtime(new FfiBackend(new NativeLibrary($library)), new RuntimeConfig(BackendType::Ffi));
$model = (new CudaBgeEmbeddingModelLoader($runtime, $library))->load($checkpoint);
$texts = [
    'short' => 'The quick brown fox jumps over lazy dogs.',
    'medium' => implode(' ', array_fill(0, 3, 'The quick brown fox jumps over lazy dogs.')),
    'long' => implode(' ', array_fill(0, 8, 'The quick brown fox jumps over lazy dogs.')),
];
$cases = [['short', 1], ['medium', 1], ['long', 1], ['short', 8], ['medium', 8], ['short', 32]];
$report = [];
foreach ($cases as [$name, $batch]) {
    $tokens = $model->tokenizer->encodeBatch(array_fill(0, $batch, $texts[$name]));
    [, $sequence] = $tokens->shape->dimensions;
    $runs = [];
    foreach ([false, true] as $graph) {
        $model->library->setGraphEnabled($graph);
        for ($warmup = 0; $warmup < 4; ++$warmup) {
            $model->library->forward($tokens->inputIds, $tokens->attentionMask->values, $tokens->tokenTypeIds, $batch, $sequence);
        }
        $times = [];
        for ($sample = 0; $sample < $samples; ++$sample) {
            $start = hrtime(true);
            $output = $model->library->forward($tokens->inputIds, $tokens->attentionMask->values, $tokens->tokenTypeIds, $batch, $sequence);
            $times[] = (hrtime(true) - $start) / 1_000;
        }
        sort($times, SORT_NUMERIC);
        $runs[$graph ? 'graph' : 'normal'] = [
            'p50_us' => percentile($times, 0.50),
            'p95_us' => percentile($times, 0.95),
            'sentences_per_second' => $batch * 1_000_000 / percentile($times, 0.50),
            'output' => $output,
        ];
    }
    $normal = $runs['normal'];
    $graph = $runs['graph'];
    $parity = compare($normal['output'], $graph['output']);
    unset($normal['output'], $graph['output']);
    $report["B{$batch}_S{$sequence}"] = [
        'normal' => $normal,
        'graph' => $graph,
        'p50_speedup' => $normal['p50_us'] / $graph['p50_us'],
        'p95_speedup' => $normal['p95_us'] / $graph['p95_us'],
        'parity' => $parity,
    ];
}
echo json_encode($report, JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR) . PHP_EOL;

/** @param list<float> $values */
function percentile(array $values, float $fraction): float
{
    return $values[(int) ceil(count($values) * $fraction) - 1];
}

/** @param list<list<float>> $expected @param list<list<float>> $actual
 * @return array{max_abs: float, mean_abs: float, max_rel: float, min_cosine: float}
 */
function compare(array $expected, array $actual): array
{
    $maxAbs = $sumAbs = $maxRel = 0.0;
    $count = 0;
    $minCosine = 1.0;
    foreach ($expected as $row => $values) {
        $dot = $left = $right = 0.0;
        foreach ($values as $column => $value) {
            $other = $actual[$row][$column];
            $error = abs($value - $other);
            $maxAbs = max($maxAbs, $error);
            $sumAbs += $error;
            ++$count;
            if (abs($value) >= 1.0e-6) {
                $maxRel = max($maxRel, $error / abs($value));
            }
            $dot += $value * $other;
            $left += $value * $value;
            $right += $other * $other;
        }
        $minCosine = min($minCosine, $dot / sqrt($left * $right));
    }

    return ['max_abs' => $maxAbs, 'mean_abs' => $sumAbs / $count, 'max_rel' => $maxRel, 'min_cosine' => $minCosine];
}
