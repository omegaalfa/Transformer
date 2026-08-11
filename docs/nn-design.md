# Neural-network architecture review (NN-R1)

Status: **approved design review; no NN implementation is authorized by this document**.

The Tensor phase T1–T10 proves native ownership, lifecycle, metadata,
materialization, and handle-to-handle execution for add, matmul, transpose, and
stable softmax. NN-R1 defines how those capabilities become neural-network
components without turning the runtime into an unplanned collection of
operators.

## Audit finding: the PHP handle bridge is still pending

The C ABI supports native Tensor handles, but the high-level PHP path does not
yet wrap them:

- `NativeLibrary` currently calls only the legacy buffer APIs;
- `NativeStorage` is a non-functional skeleton;
- PHP `Tensor` construction, metadata, and operations are skeletons;
- no PHP object currently guarantees exactly-once destruction of a native
  Tensor handle.

Therefore, the proven path today is native C ABI handle-to-handle execution,
not yet an end-to-end high-level PHP Tensor chain. A focused **NN-0 native
Tensor bridge** must close this gap before `Linear` can execute from PHP. This
is integration of the completed Tensor ABI, not a new numerical kernel.

## Placement of responsibilities

The architecture is deliberately split:

```text
PHP
  model composition
  Module and Parameter metadata
  parameter names and state dictionaries
  validation and user-facing exceptions
  tokenizer output and model configuration

Rust
  Tensor storage and lifecycle
  shape/dtype invariants
  numerical execution
  operation-specific validation at the C ABI
  newly allocated result Tensors
```

`Linear`, `Embedding`, `LayerNorm`, `GELU`, and later Transformer blocks are
PHP composition objects. Their numerical work stays in Rust. They must never
copy native Tensor data to PHP between operations.

## Parameter contract

`Parameter` is not a second Tensor implementation. It wraps one PHP `Tensor`
whose storage may own a native handle, plus model metadata:

```text
Parameter
  name: local module name
  tensor: Tensor
  trainable: bool
```

Rules for v1:

- shape, strides, dtype, device, and storage exist only in Tensor;
- Parameter never duplicates or mutates Tensor storage;
- `trainable` is descriptive metadata; v1 is inference-only;
- there is no gradient, optimizer, autograd graph, or training state;
- parameters may be read by multiple modules because Tensor operations are
  immutable, while native handle destruction remains owned by `NativeStorage`;
- module traversal produces qualified names such as `encoder.layer0.weight`;
  Parameter stores only its local name.

## Module contract

The current `Module::forward(Tensor): Tensor` signature is too restrictive as a
universal interface. Embedding begins with token IDs, and later attention also
accepts masks. NN-1 should make `Module` the introspection/composition contract:

```text
parameters() -> map<string, Parameter>
modules()    -> map<string, Module>
```

Concrete modules declare their own typed `forward` method. Tensor-to-Tensor
components such as `Linear`, `LayerNorm`, and `GELU` retain
`forward(Tensor): Tensor`. Explicit maps are preferred over reflection or magic
property discovery. Recursive traversal builds a deterministic state dict.

Leaf modules receive a `Runtime` explicitly and dispatch through its backend.
Inputs, parameters, and results must use compatible backend, device, and dtype.
The PHP objects know neither FFI pointers nor Rust layouts.

## Native handle ownership in PHP

NN-0 must introduce one PHP owner for each native handle, conceptually
`NativeStorage`:

- construction receives a live opaque handle;
- copying a PHP object reference does not duplicate the handle;
- its destructor calls native destroy at most once;
- a released/moved handle cannot be used again;
- operation results create new `NativeStorage` owners;
- metadata is cached only if it is immutable and verified from the ABI;
- `toArray()` is the only explicit data materialization boundary.

Backend operations accept PHP Tensors and delegate native-handle operations to
`NativeLibrary`. No intermediate PHP arrays are allowed in an NN forward pass.

## Module dependency map

### Linear

```text
Linear
  weight Parameter
  optional bias Parameter
  native last-dimension projection
```

The current strict Tensor add cannot add bias `[out_features]` to
`[rows, out_features]`, and current matmul accepts rank 2 only. Transformer
inputs will eventually have leading dimensions such as
`[batch, sequence, hidden]`. NN-2 should therefore design one specific native
Linear operation that:

- requires the input's last dimension to equal `input_features`;
- treats preceding dimensions as rows without exposing a PHP copy;
- multiplies by weight shaped `[input_features, output_features]`;
- adds optional bias shaped `[output_features]` inside the same native path;
- preserves all leading dimensions and replaces only the last one;
- returns a new contiguous Tensor.

This avoids prematurely adding general broadcasting or reshape/view semantics.
The reference tests still compare its matmul portion with the naive kernel.

### Embedding

```text
Embedding
  weight Parameter [vocabulary_size, dimensions]
  token IDs
  gather rows
```

Token IDs must not be encoded as Float32. Because tokenizer output originates
in PHP and Tensor v1 intentionally supports only Float32, NN-3 should initially
define a dedicated embedding lookup boundary accepting validated integer token
IDs and a Float32 weight handle. It returns a native Float32 Tensor and performs
only one PHP-to-native transfer at the model input boundary. A general integer
Tensor dtype is deferred until another real consumer justifies it.

Validation includes non-negative IDs, vocabulary bounds, output shape, and
integer-width conversion. A general-purpose gather operator is not required
before this contract is proven.

### LayerNorm

```text
LayerNorm
  input Tensor
  gamma Parameter
  beta Parameter
  epsilon
  stable normalization over last dimension
```

Rather than first exposing unrelated public mean, variance, sub, mul, and
rsqrt operators, NN-4 should begin with a safe reference LayerNorm kernel. It
computes mean and variance stably over the last dimension, applies epsilon,
gamma, and beta, preserves leading dimensions, and returns a new Tensor.
Primitive kernels should be extracted later only when multiple consumers need
them.

### GELU

NN-5 introduces one explicit Float32 GELU reference kernel and Tensor-handle
operation. The review must choose and document exact versus tanh approximation
before implementation; model configuration/weights must use the matching
variant. GELU is element-wise, immutable, shape-preserving, and requires parity
against a trusted reference.

## Gates

```text
NN-R1  architecture review                              COMPLETE
NN-0   PHP Native Tensor bridge                         PENDING
NN-1   Parameter + Module introspection contracts       PENDING
NN-2   Linear with native last-dimension projection     PENDING
NN-3   Embedding with validated integer lookup          PENDING
NN-4   stable LayerNorm                                  PENDING
NN-5   GELU                                              PENDING
NN-R2  review before Attention                          PENDING
```

Every implementation gate requires its own approval, tests, and documentation
update. NN-R2 must review Q/K/V orientation, batch/sequence/head shapes,
masking, scaling, and softmax axis before any Attention implementation.

## Explicitly deferred

- training, gradients, autograd, and optimizers;
- generic broadcasting;
- generic gather before Embedding proves its requirements;
- generic reductions solely to assemble LayerNorm;
- views and non-contiguous Tensors;
- additional dtypes solely to represent tokenizer output;
- SIMD, BLAS, GPU, and fused Transformer blocks;
- attention implementation before NN-R2.

