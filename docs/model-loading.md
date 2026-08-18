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

## Deferred

- F16/BF16 conversion and quantization;
- sharded Safetensors and index files;
- downloading, caching or remote trust policy;
- memory mapping and loading-performance work.

No Graph Executor or NN-1 through NN-7 behavior is changed. The native ABI was
extended only with isolated ExactGELU and BERT-attention symbols.
