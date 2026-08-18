# Model loading

Model loading is intentionally split into small gates. Completing one gate does
not imply that a real model or every Safetensors dtype is executable.

```text
MODEL-R2A  file header -> validated tensor descriptors       complete
MODEL-R2B  named descriptor -> exact payload bytes           complete
MODEL-R2C  Float32 payload -> independent runtime Tensor      complete
MODEL-R2D  closed checkpoint manifest -> resident Parameters complete
MODEL-R2E  config + Parameters -> atomic BertModel            complete
```

NN-1 through NN-7 remain an immutable Pre-Norm family. The BERT-compatible
encoder is additive and uses its own Post-Norm blocks, biased attention
projections and exact GELU contract.

## Safetensors structure

`SafetensorsReader` reads the little-endian 64-bit header length, parses the
JSON object and returns a `WeightMap`. A `TensorMetadata` contains the tensor
name, shape, dtype, absolute file offset and payload byte length. Header size,
JSON structure, shapes, dtype sizes, offsets and complete payload coverage are
validated before metadata is returned.

The reader recognizes `F32`, `F16`, `BF16`, `I64` and `I8` as file metadata.
This does not mean that the runtime executes all five. `I64` is needed for the
checkpoint's auxiliary `position_ids`; it is never materialized as a runtime
Tensor. `tensor(path, name)` reads one
validated range as unchanged bytes and does not create a Tensor.

## Float32 materialization

`WeightMaterializer` currently accepts only `F32`. Every other dtype is
rejected rather than converted silently. It validates expected checkpoint
shape and byte length, decodes IEEE-754 little-endian Float32 values, rejects
NaN and infinities, applies an explicitly declared orientation and creates one
independent Tensor through the selected backend.

Supported orientations are:

```text
Identity
PyTorchLinearTranspose: checkpoint [out,in] -> runtime [in,out]
```

Square Linear weights are transposed as values too; equal shapes never imply
identity conversion.

## Closed weight manifests

`WeightManifest` is an ordered list of `CheckpointParameterSpec` entries. Each
entry declares:

```text
checkpoint tensor name
runtime Parameter name
checkpoint shape
runtime shape
orientation
```

`SafetensorsWeightLoader` rejects missing tensors, unexpected tensors,
duplicate checkpoint names, duplicate Parameter names and shape mismatches.
The complete structural manifest is validated before any payload is
materialized. It returns a Parameter map only after every entry succeeds;
Parameters are resident, independent and non-trainable.

The BGE manifest contains 197 resident Parameters and explicitly classifies
`embeddings.position_ids` plus the two pooler tensors as non-encoder tensors.
The loader accepts only the union of Parameters and this named ignore set;
every other extra tensor is rejected.

## BERT construction

`BertConfigReader` validates the BERT configuration needed by the target.
`BertModelLoader` then validates the target architecture, validates the closed
manifest, materializes every Parameter, constructs embeddings and all 12
Post-Norm blocks privately, and returns the model only after the complete tree
succeeds. No partially constructed model is published.

The executable output contract is `lastHiddenState [B,S,D]`. Embeddings combine
word, derived absolute position and token-type tables before LayerNorm. Each
block executes biased bidirectional attention followed by Post-Norm, then a
biased FFN with exact-erf GELU followed by Post-Norm.

## BGE sentence embeddings

`BgeEmbeddingModelLoader` composes the unchanged `BertModel` with the official
WordPiece vocabulary from `tokenizer.json`. Tokenization produces row-major
`input_ids`, `attention_mask` and `token_type_ids`, truncates at the configured
512-token limit, and pads batches to their longest sequence.

`BgeEmbeddingModel::encodeBatch()` applies the checkpoint's official CLS
pooling: it selects `lastHiddenState[b,0,:]` for every batch row and then
L2-normalizes each `[D]` vector. Padding cannot change which token is selected.
A pooled row with zero norm is an explicit error and produces no output Tensor.
For `BAAI/bge-small-en-v1.5`, the final output is `[B,384]`. Direct `BertModel`
execution remains available for callers that require `lastHiddenState`.

`BgePoolingStrategy::Cls` is the default. `BgePoolingStrategy::Mean` explicitly
selects the independent generic `MeanPooling` component, which averages only
positions enabled by the attention mask. Mean mode is not presented as the
official BGE-small-en-v1.5 pooling policy.

### Public execution example

```php
use Omegaalfa\Transformer\Backend\BackendType;
use Omegaalfa\Transformer\Backend\Ffi\FfiBackend;
use Omegaalfa\Transformer\Backend\Ffi\NativeLibrary;
use Omegaalfa\Transformer\Model\Loader\BgeEmbeddingModelLoader;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Runtime\RuntimeConfig;

$native = new NativeLibrary(NativeLibrary::defaultPath(__DIR__));
$runtime = new Runtime(
    new FfiBackend($native),
    new RuntimeConfig(BackendType::Ffi),
);

$model = (new BgeEmbeddingModelLoader($runtime))->load(
    __DIR__ . '/models/bge-small-en-v1.5',
);

$one = $model->encode('hello world'); // list<float>, 384 values
$many = $model->encodeBatch([
    'hello world',
    'A second sentence with a different length.',
]); // list<list<float>>, logical shape [2,384]

$details = $model->encodeBatchOutput(['hello world']);
$pooled = $details->pooled;       // Tensor [1,384], CLS before L2
$normalized = $details->embedding; // Tensor [1,384], unit-norm row
```

The loader is intended to be called once for a long-lived application
lifecycle. Each `encode` call creates independent output Tensors while the 197
model Parameters remain resident and immutable.

The public `encode()` facade returns one flat PHP `list<float>` and
`encodeBatch()` returns `list<list<float>>`. Tokenization, BERT inputs,
attention masks, pooling, normalization and intermediate Tensors are hidden.
Load once and encode many times in a persistent CLI process or worker to retain
all model Parameters; no server integration is implied.

### End-to-end benchmark

The single-process benchmark measures cold load, warm single encode,
short/medium/long batches, resident versus recreate, and RSS:

```bash
TRANSFORMER_BGE_CHECKPOINT=/path/to/bge-small-en-v1.5 \
TRANSFORMER_BGE_SAMPLES=15 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
php -d xdebug.mode=off runtime/benches/bge_embedding.php
```

To request the non-official generic mean strategy explicitly:

```php
use Omegaalfa\Transformer\Embedding\BgePoolingStrategy;

$meanModel = (new BgeEmbeddingModelLoader($runtime))->load(
    __DIR__ . '/models/bge-small-en-v1.5',
    BgePoolingStrategy::Mean,
);
```

## Deferred

- F16/BF16 conversion and quantization;
- sharded Safetensors and index files;
- downloading, caching or remote trust policy;
- memory mapping and loading-performance work.

No Graph Executor or NN-1 through NN-7 behavior is changed. The native ABI was
extended only with isolated ExactGELU and BERT-attention symbols.
