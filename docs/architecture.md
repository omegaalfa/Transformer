# Architecture

The project has three boundaries:

1. The PHP API contains Tensor, neural-network, Transformer, tokenizer, model,
   embedding, serialization, and future generation contracts.
2. `BackendInterface` isolates those APIs from execution. High-level modules do
   not know about FFI, Zend, raw pointers, or Rust internals.
3. The separately built Rust runtime owns native execution. The FFI backend and
   future Zend extension consume the same stable C ABI.

```text
PHP high-level API
        |
Backend abstraction
        |
C ABI
        |
Rust runtime built by Cargo
```

## Responsibilities

- **Tensor** describes shape, dtype, device, storage, and future operations.
- **Backend** defines execution contracts and selects an implementation.
- **NN/Transformer/Model** compose typed high-level behavior without depending
  on native implementation details.
- **NativeLibrary** owns PHP FFI loading and C buffer calls.
- **Rust FFI module** validates pointer-level arguments, prevents panics from
  crossing the ABI, and converts raw pointers into slices.
- **Rust kernels** contain safe computation over slices and no raw pointers.

The Float32 Tensor handle API now provides addition, naive matmul, materialized
transpose, and stable 1D softmax. The high-level PHP Tensor/NativeStorage bridge
is still pending; legacy buffer calls remain available as numerical references.

The proposed Tensor v1 boundaries, invariants, ownership model, opaque-handle
direction, and staged implementation gates are documented in
[`native-tensor-design.md`](native-tensor-design.md). That document is a design
record of the completed T1–T10 phase. The following NN boundary and module
decisions are documented in [`nn-design.md`](nn-design.md).

## Mandatory native-kernel rule

> Kernels never receive raw pointers. Raw pointers exist only at the FFI
> boundary.

Every future operation follows the same direction:

```text
raw C pointers
    -> runtime/src/ffi
    -> validated lengths and safe slices
    -> runtime/src/kernels
```

This applies to future reference implementations of matmul, transpose,
softmax, normalization, and attention. Optimization does not weaken this
boundary.
