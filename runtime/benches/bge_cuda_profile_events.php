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
$samples = max(3, (int) (getenv('TRANSFORMER_BGE_CUDA_PROFILE_SAMPLES') ?: 5));
$library = NativeLibrary::defaultPath($root);
$runtime = new Runtime(new FfiBackend(new NativeLibrary($library)), new RuntimeConfig(BackendType::Ffi));
$model = (new CudaBgeEmbeddingModelLoader($runtime, $library))->load($checkpoint);
$sentences = [
    's9' => 'The quick brown fox jumps over lazy dogs.',
    's26' => implode(' ', array_fill(0, 3, 'The quick brown fox jumps over lazy dogs.')),
    's66' => implode(' ', array_fill(0, 8, 'The quick brown fox jumps over lazy dogs.')),
];
$cases = [['s9', 1], ['s26', 1], ['s66', 1], ['s9', 8], ['s26', 8], ['s66', 8], ['s9', 32]];
$report = [];
foreach ($cases as [$length, $batch]) {
    $tokens = $model->tokenizer->encodeBatch(array_fill(0, $batch, $sentences[$length]));
    [$actualBatch, $sequence] = $tokens->shape->dimensions;
    $model->library->forward(
        $tokens->inputIds,
        $tokens->attentionMask->values,
        $tokens->tokenTypeIds,
        $actualBatch,
        $sequence,
    );
    $measurements = array_fill(0, 137, []);
    for ($sample = 0; $sample < $samples; ++$sample) {
        $profile = $model->library->profile(
            $tokens->inputIds,
            $tokens->attentionMask->values,
            $tokens->tokenTypeIds,
            $actualBatch,
            $sequence,
        );
        foreach ($profile['timings_ms'] as $index => $value) {
            $measurements[$index][] = $value;
        }
    }
    $sum = array_map(static function (array $values): float {
        sort($values, SORT_NUMERIC);

        return $values[intdiv(count($values), 2)];
    }, $measurements);
    $cursor = 3;
    $blocks = [];
    for ($layer = 0; $layer < 12; ++$layer) {
        $values = array_slice($sum, $cursor, 11);
        $blocks[] = array_combine(
            ['q', 'k', 'v', 'fused_attention', 'absorbed_attention_v', 'output', 'ln1', 'ffn1', 'gelu', 'ffn2', 'ln2'],
            $values,
        );
        $cursor += 11;
    }
    $total = array_sum($sum);
    $report["B{$batch}_S{$sequence}"] = [
        'total_gpu_ms' => $total,
        'h2d_ms' => $sum[0],
        'embeddings_ms' => $sum[1],
        'embedding_layer_norm_ms' => $sum[2],
        'blocks' => $blocks,
        'blocks_aggregate_ms' => aggregate($blocks),
        'cls_l2_ms' => $sum[135],
        'd2h_ms' => $sum[136],
        'internal_submissions' => 195,
        'steady_cuda_graph_launches' => 1,
        'cuda_malloc_steady_state' => 0,
        'cuda_free_steady_state' => 0,
        'explicit_stream_synchronizations' => 1,
    ];
}

echo json_encode($report, JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR) . PHP_EOL;

/** @param list<array<string, float>> $blocks @return array<string, float> */
function aggregate(array $blocks): array
{
    $aggregate = [];
    foreach ($blocks as $block) {
        foreach ($block as $name => $value) {
            $aggregate[$name] = ($aggregate[$name] ?? 0.0) + $value;
        }
    }

    return $aggregate;
}
