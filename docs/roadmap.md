# Roadmap

- [ ] **Milestone A** — `PurePhpBackend` and basic Tensor.
- [ ] **Milestone B** — Linear, Embedding, normalization, and Attention in PHP.
- [x] **Milestone C foundation** — Rust runtime and first kernel through FFI.
- [ ] **Milestone D** — softmax, normalization, and attention in Rust.
- [ ] **Milestone E** — Zend extension consuming the Rust C ABI.
- [ ] **Milestone F** — real encoder and Safetensors (validated metadata,
  selective payload reading, Float32 materialization and closed Parameter
  manifests complete; BERT config/model construction pending).
- [ ] **Milestone G** — ContextEngine integration.
- [ ] **Milestone H** — CPU optimization after profiling (tiled matmul and optional BLAS complete; SIMD and threads pending).
- [ ] **Milestone I** — decoder and KV cache.
- [ ] **Milestone J** — quantization and GPU.

The intended future integration remains:

```text
ContextEngine
    -> NativeTransformerEmbeddingProvider
        -> Rust runtime
            -> encoder
                -> embedding
```

The scalar `matmul_f32` reference remains available. Cache-friendly and tiled
Rust kernels plus an optional OpenBLAS SGEMM backend are selected by an isolated
production dispatcher for benchmarked Transformer shapes.

## Native runtime detail

- [x] Rust `cdylib` built by Cargo.
- [x] Stable C ABI consumed by PHP FFI.
- [x] Native version smoke test.
- [x] Float32 buffers crossing PHP, FFI, Rust, and back.
- [x] Safe `add_f32` kernel and isolated unsafe boundary.
- [x] Panic containment.
- [x] Rust unit tests and PHP integration tests.
- [x] Naive reference `matmul_f32`.
- [x] `transpose_f32` with explicit row-major shape/layout mapping.
- [x] Numerically stable one-dimensional `softmax_f32`.
- [x] Native Tensor and opaque handles.
- [x] Native Shape, strides, dtype, and storage.
- [x] Rank-N softmax over the last dimension.
- [x] Resident PHP Tensor/NativeStorage bridge.
- [x] Linear, Embedding, LayerNorm, GELU, non-causal MultiHeadAttention, FeedForward, and Pre-Norm TransformerBlock execution.

The initial `matmul_f32` must be a straightforward safe reference kernel. SIMD,
Rayon, BLAS, GPU execution, and other optimizations come only after known-output
tests and parity establish correctness.

## Architectural review gate

Kernel expansion stops here. Before another operator is added, review and
decide:

- whether Tensor owns its buffer and whether views are supported;
- Shape and stride representation;
- whether transpose copies data or changes strides;
- dtype representation;
- separation of storage and Tensor;
- opaque handle design and destruction ownership.

The current proposal is recorded in
[`native-tensor-design.md`](native-tensor-design.md). Implementation is split
into gates T1–T10; all ten gates are complete.

- [x] **T1** — DType, Shape, and contiguous row-major Strides in pure Rust.
- [x] **T2** — exclusively owned contiguous `Vec<f32>` Storage.
- [x] **T3** — contiguous CPU Float32 Tensor.
- [x] **T4** — opaque C ABI handle and lifecycle contract.
- [x] **T5** — create, destroy, shape, rank, and dtype ABI.
- [x] **T6** — explicit numel query and copy-out for debugging.
- [x] **T7** — Tensor add with buffer-API parity.
- [x] **T8** — Tensor matmul with reference parity.
- [x] **T9** — materialized contiguous Tensor transpose.
- [x] **T10** — stable Tensor softmax with buffer-API parity.

The NN architecture review is complete and did not add numerical code:

- [x] **NN-R1** — Parameter, Module, PHP/Rust boundary, and dependency design.
- [x] **NN-0** — high-level PHP bridge for native Tensor handles.
- [x] **NN-1** — Parameter and Module introspection contracts.
- [x] **NN-2** — Linear with native last-dimension projection and bias.
- [x] **NN-3** — Embedding with validated integer lookup.
- [x] **NN-4** — stable inference LayerNorm with resident gamma/beta.
- [x] **NN-5** — Float32 tanh GELU with Float64 internal calculation.
- [x] **NN-R2** — architecture review before Attention.
- [x] **NN-6** — non-causal MultiHeadAttention with boolean padding mask.
- [x] **NN-R3** — architecture review before TransformerBlock.
- [x] **NN-7** — resident FeedForward and Pre-Norm TransformerBlock composition.

The model-loading foundation is also staged independently:

- [x] **MODEL-R2A** — validated Safetensors header and tensor descriptors.
- [x] **MODEL-R2B** — selective exact-byte tensor payload reading.
- [x] **MODEL-R2C** — strict Float32 Tensor materialization and documented
  PyTorch Linear transposition.
- [x] **MODEL-R2D** — closed manifest mapping to resident Parameters.
- [ ] **MODEL-R2E** — config validation and atomic BERT-compatible model
  construction.

See [`nn-design.md`](nn-design.md) for the decisions and deferred scope.
See [`model-loading.md`](model-loading.md) for the loader contracts and current
dtype policy.
