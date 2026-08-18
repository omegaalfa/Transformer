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
$samples = max(5, (int) (getenv('TRANSFORMER_BGE_CUDA_MODE_SAMPLES') ?: 15));
$library = NativeLibrary::defaultPath($root);
$runtime = new Runtime(new FfiBackend(new NativeLibrary($library)), new RuntimeConfig(BackendType::Ffi));
$model = (new CudaBgeEmbeddingModelLoader($runtime, $library))->load($checkpoint);
$sentences = [
    's9' => 'The quick brown fox jumps over lazy dogs.',
    's26' => implode(' ', array_fill(0, 3, 'The quick brown fox jumps over lazy dogs.')),
    's66' => implode(' ', array_fill(0, 8, 'The quick brown fox jumps over lazy dogs.')),
];
$result = [];
foreach ($sentences as $name => $sentence) {
    foreach ([1, 8, 32] as $batch) {
        $texts = array_fill(0, $batch, $sentence);
        $tokens = count($model->tokenizer->encode($sentence)->tokenIds);
        $modeOutputs = [];
        foreach (['fp32' => false, 'tf32' => true] as $mode => $enabled) {
            $model->library->setTensorFloat32($enabled);
            for ($warmup = 0; $warmup < 3; ++$warmup) {
                $model->encodeBatch($texts);
            }
            $times = [];
            for ($sample = 0; $sample < $samples; ++$sample) {
                $start = hrtime(true);
                $modeOutputs[$mode] = $model->encodeBatch($texts);
                $times[] = (hrtime(true) - $start) / 1_000;
            }
            sort($times, SORT_NUMERIC);
            $result[$name][(string) $batch][$mode] = [
                'tokens' => $tokens,
                'p50_us' => percentile($times, 0.50),
                'p95_us' => percentile($times, 0.95),
                'p99_us' => percentile($times, 0.99),
                'sentences_s' => $batch * 1_000_000 / percentile($times, 0.50),
                'tokens_s' => $batch * $tokens * 1_000_000 / percentile($times, 0.50),
            ];
        }
        $result[$name][(string) $batch]['tf32_vs_fp32'] = metrics($modeOutputs['tf32'], $modeOutputs['fp32']);
    }
}

echo json_encode($result, JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR) . PHP_EOL;

/** @param list<float> $sorted */
function percentile(array $sorted, float $fraction): float
{
    return $sorted[(int) ceil(count($sorted) * $fraction) - 1];
}

/** @param list<list<float>> $actual @param list<list<float>> $expected
 *  @return array{max_abs: float, mean_abs: float, max_rel: float, min_cosine: float, max_norm_delta: float}
 */
function metrics(array $actual, array $expected): array
{
    $maxAbs = $sumAbs = $maxRel = $normDelta = 0.0;
    $minimumCosine = 1.0;
    $count = 0;
    foreach ($expected as $row => $values) {
        $dot = $actualNorm = $expectedNorm = 0.0;
        foreach ($values as $column => $value) {
            $candidate = $actual[$row][$column];
            $error = abs($candidate - $value);
            $maxAbs = max($maxAbs, $error);
            $sumAbs += $error;
            ++$count;
            if (abs($value) >= 1.0e-3) {
                $maxRel = max($maxRel, $error / abs($value));
            }
            $dot += $candidate * $value;
            $actualNorm += $candidate * $candidate;
            $expectedNorm += $value * $value;
        }
        $actualNorm = sqrt($actualNorm);
        $expectedNorm = sqrt($expectedNorm);
        $minimumCosine = min($minimumCosine, $dot / ($actualNorm * $expectedNorm));
        $normDelta = max($normDelta, abs($actualNorm - $expectedNorm));
    }

    return [
        'max_abs' => $maxAbs,
        'mean_abs' => $sumAbs / $count,
        'max_rel' => $maxRel,
        'min_cosine' => $minimumCosine,
        'max_norm_delta' => $normDelta,
    ];
}
