<?php

declare(strict_types=1);

use Omegaalfa\Transformer\Backend\BackendType;
use Omegaalfa\Transformer\Backend\Ffi\FfiBackend;
use Omegaalfa\Transformer\Backend\Ffi\NativeLibrary;
use Omegaalfa\Transformer\Embedding\Normalization\L2Normalizer;
use Omegaalfa\Transformer\Embedding\Pooling\ClsPooling;
use Omegaalfa\Transformer\Model\Encoder\Bert\BertModelInput;
use Omegaalfa\Transformer\Model\Loader\BgeEmbeddingModelLoader;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Runtime\RuntimeConfig;
use Omegaalfa\Transformer\Tensor\Shape;

require dirname(__DIR__, 2) . '/vendor/autoload.php';

$root = dirname(__DIR__, 2);
$checkpoint = getenv('TRANSFORMER_BGE_CHECKPOINT') ?: '/tmp/transformer-model-r3/bge-small-en-v1.5';
$samples = max(3, (int) (getenv('TRANSFORMER_BGE_PROFILE_SAMPLES') ?: 5));
$runtime = new Runtime(
    new FfiBackend(new NativeLibrary(NativeLibrary::defaultPath($root))),
    new RuntimeConfig(BackendType::Ffi),
);
$pipeline = (new BgeEmbeddingModelLoader($runtime))->load($checkpoint);
$texts = [
    'short' => 'How do I request annual leave?',
    'medium' => implode(' ', array_fill(0, 2, 'The employee handbook explains the approval process and expected response time.')),
    'long' => implode(' ', array_fill(0, 4, 'This representative document sentence contains enough ordinary English words to exercise a longer BERT sequence.')),
];

$results = [];
foreach ($texts as $name => $text) {
    $totals = [];
    $tokens = $pipeline->tokenizer->encodeBatch([$text]);
    $input = new BertModelInput($tokens->inputIds, $tokens->shape, $tokens->attentionMask, $tokens->tokenTypeIds);
    for ($sample = 0; $sample < $samples; ++$sample) {
        $started = hrtime(true);
        $hidden = timed($totals, 'embeddings', fn () => $pipeline->model->embeddings->forward($input->inputIds, $input->shape, $input->tokenTypeIds));
        foreach ($pipeline->model->layers as $index => $layer) {
            $blockStart = hrtime(true);
            $attention = timed($totals, 'attention', fn () => $layer->attention->forward($hidden, $input->attentionMask));
            $afterAttention = timed($totals, 'attention_residual_norm', fn () => $layer->attentionNorm->forward($hidden->add($attention)));
            $ffnInput = timed($totals, 'ffn_input_linear', fn () => $layer->feedForward->inputProjection->forward($afterAttention));
            $activated = timed($totals, 'exact_gelu', fn () => $layer->feedForward->activation->forward($ffnInput));
            $ffnOutput = timed($totals, 'ffn_output_linear', fn () => $layer->feedForward->outputProjection->forward($activated));
            $hidden = timed($totals, 'ffn_residual_norm', fn () => $layer->feedForwardNorm->forward($afterAttention->add($ffnOutput)));
            $totals['block_' . $index] = ($totals['block_' . $index] ?? 0.0) + elapsedUs($blockStart);
        }
        $pooled = timed($totals, 'pooling', fn () => (new ClsPooling($runtime))->pool($hidden));
        $normalized = timed($totals, 'l2_normalization', fn () => (new L2Normalizer($runtime))->normalize($pooled));
        timed($totals, 'php_materialization', fn () => $normalized->toFloat32());
        $totals['total'] = ($totals['total'] ?? 0.0) + elapsedUs($started);
    }
    foreach ($totals as $stage => $microseconds) {
        $totals[$stage] = $microseconds / $samples;
    }
    $total = $totals['total'];
    $percentages = [];
    foreach ($totals as $stage => $microseconds) {
        $percentages[$stage] = $microseconds / $total * 100.0;
    }
    $results[$name] = [
        'tokens' => count($tokens->inputIds),
        'mean_us' => $totals,
        'percent' => $percentages,
    ];
}

echo json_encode([
    'samples' => $samples,
    'structural_counts' => [
        'ffi_calls' => 426,
        'result_handles_created' => 105,
        'result_handles_destroyed' => 105,
        'php_intermediate_materializations' => 2,
        'minimum_rust_vec_allocations' => 177,
    ],
    'profiles' => $results,
], JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR) . PHP_EOL;

/** @template T
 * @param array<string, float> $totals
 * @param Closure(): T $operation
 * @return T
 */
function timed(array &$totals, string $stage, Closure $operation): mixed
{
    $start = hrtime(true);
    $result = $operation();
    $totals[$stage] = ($totals[$stage] ?? 0.0) + elapsedUs($start);

    return $result;
}

function elapsedUs(int $start): float
{
    return (hrtime(true) - $start) / 1_000;
}
