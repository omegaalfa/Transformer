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
$iterations = max(2_000, (int) (getenv('TRANSFORMER_BGE_HARDENING_ITERATIONS') ?: 2_000));
$library = NativeLibrary::defaultPath($root);
$runtime = new Runtime(new FfiBackend(new NativeLibrary($library)), new RuntimeConfig(BackendType::Ffi));
$models = [];
foreach (CudaBgePrecision::cases() as $precision) {
    $models[$precision->name] = (new CudaBgeEmbeddingModelLoader($runtime, $library, $precision))->load($checkpoint);
}
$texts = ['short' => 'hello world',
    'medium' => implode(' ', array_fill(0, 3, 'The quick brown fox jumps over the lazy dog.')),
    'long' => implode(' ', array_fill(0, 8, 'The quick brown fox jumps over the lazy dog.'))];
$parity = [];
foreach ($texts as $name => $text) {
    $tokens = $models['Float32']->tokenizer->encodeBatch([$text]);
    [$batch, $sequence] = $tokens->shape->dimensions;
    $states = [];
    foreach ($models as $precision => $model) {
        $states[$precision] = $model->library->benchmarkStates($tokens->inputIds,
            $tokens->attentionMask->values, $tokens->tokenTypeIds, $batch, $sequence);
    }
    foreach (['Float16', 'BFloat16'] as $precision) {
        foreach ($states['Float32']['states'] as $state => $expected) {
            $parity[$name][$precision]['state_' . $state] = compare($expected, $states[$precision]['states'][$state]);
        }
        $parity[$name][$precision]['invalid'] = $states[$precision]['invalid'];
    }
}

$problem = array_fill(0, 8, $texts['long']);
$tokens = $models['Float16']->tokenizer->encodeBatch($problem);
[$batch, $sequence] = $tokens->shape->dimensions;
$soak = [];
foreach (['Float16', 'BFloat16'] as $precision) {
    $model = $models[$precision];
    foreach ([false, true] as $graph) {
        $model->library->setGraphEnabled($graph);
        for ($index = 0; $index < 100; ++$index) {
            $model->encodeBatch($problem);
        }
        $first = $model->encodeBatch($problem);
        $firstInvalid = null;
        $nan = $positiveInf = $negativeInf = 0;
        $maxDrift = 0.0;
        $bitwise = 0;
        for ($index = 1; $index <= $iterations; ++$index) {
            $actual = $model->encodeBatch($problem);
            $bitwise += $actual === $first ? 1 : 0;
            $maxDrift = max($maxDrift, drift($first, $actual));
            $invalid = $model->library->benchmarkFinite($tokens->inputIds,
                $tokens->attentionMask->values, $tokens->tokenTypeIds, $batch, $sequence);
            if ($invalid['category'] !== 0) {
                $nan += $invalid['category'] === 1 ? 1 : 0;
                $positiveInf += $invalid['category'] === 2 ? 1 : 0;
                $negativeInf += $invalid['category'] === 3 ? 1 : 0;
                if ($firstInvalid === null) {
                    $firstInvalid = ['iteration' => $index, 'category' => $invalid['category'],
                        'stage' => $invalid['stage'], 'layer' => $invalid['layer'],
                        'index' => $invalid['index'], 'value' => match ($invalid['category']) {
                            1 => 'NaN', 2 => '+Inf', 3 => '-Inf', default => 'finite',
                        }];
                }
            }
        }
        $soak[$precision][$graph ? 'graph_on' : 'graph_off'] = [
            'iterations' => $iterations, 'bitwise_identical' => $bitwise, 'max_drift' => $maxDrift,
            'nan' => $nan, 'positive_inf' => $positiveInf, 'negative_inf' => $negativeInf,
            'first_invalid' => $firstInvalid, 'diagnostics' => $model->library->benchmarkDiagnostics(),
        ];
    }
}

echo json_encode(['sequence' => $sequence, 'parity' => $parity, 'soak' => $soak],
    JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR) . PHP_EOL;

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
        'cosine' => $dot / sqrt($left * $right)];
}

/** @param list<list<float>> $expected @param list<list<float>> $actual */
function drift(array $expected, array $actual): float
{
    $max = 0.0;
    foreach ($expected as $row => $values) {
        foreach ($values as $column => $value) {
            $max = max($max, abs($actual[$row][$column] - $value));
        }
    }

    return $max;
}
