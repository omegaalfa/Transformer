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
$model = (new CudaBgeEmbeddingModelLoader($runtime, $library, CudaBgePrecision::Float16))->load($checkpoint);
$texts = array_fill(0, 8, implode(' ', array_fill(0, 8, 'The quick brown fox jumps over the lazy dog.')));
$tokens = $model->tokenizer->encodeBatch($texts);
[$batch, $sequence] = $tokens->shape->dimensions;
$model->library->setGraphEnabled(false);
$model->encodeBatch($texts);
$invalid = $model->library->benchmarkFinite($tokens->inputIds, $tokens->attentionMask->values,
    $tokens->tokenTypeIds, $batch, $sequence);
$model->library->setGraphEnabled(true);
$model->encodeBatch($texts);
$model->encodeBatch($texts);
$output = $model->encodeBatch($texts);
echo json_encode(['batch' => $batch, 'sequence' => $sequence, 'invalid' => $invalid,
    'finite_output' => count($output) === 8], JSON_THROW_ON_ERROR) . PHP_EOL;
