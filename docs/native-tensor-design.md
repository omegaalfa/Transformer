# Native Tensor design proposal

Status: **T1 through T10 implemented; NN-R1 complete; PHP handle bridge is the next gate**.

The validated buffer APIs established the C ABI, PHP FFI, safe-kernel boundary,
row-major indexing, dimensional validation, and numerical behavior. They are
not the final execution model because every call currently copies data between
PHP arrays and native buffers.

## Objective

Keep tensor data in Rust across operation chains:

```text
PHP Tensor
    |
    +-- opaque handle
            |
            v
       Rust Tensor
            +-- add
            +-- matmul
            +-- transpose
            +-- softmax
```

Only an explicit debugging/materialization operation copies data back to PHP.

## Tensor v1 boundaries

The first native Tensor is deliberately conservative:

- CPU only;
- Float32 only;
- contiguous row-major storage only;
- Tensor exclusively owns its storage;
- no views or shared storage;
- no broadcasting;
- no implicit dtype conversion;
- no in-place public operations;
- operations return newly owned tensors.

These restrictions reduce simultaneous uncertainty while leaving clear
extension points for later milestones.

## Responsibilities

Conceptually, the model is:

```rust
pub struct Tensor {
    storage: Storage,
    shape: Shape,
    strides: Strides,
    dtype: DType,
}
```

This is the committed T3 structure in `runtime/src/tensor/tensor.rs`.

### DType

Describes how storage elements are interpreted. Tensor v1 supports only
`Float32`, but dtype remains explicit so Tensor does not become permanently
coupled to one representation.

### Shape

Owns an ordered list of dimension sizes. It is responsible for checked element
count calculation and rejects dimension products that overflow `usize`.

### Strides

Maps logical coordinates to storage offsets. Contiguous row-major strides are
derived from Shape using checked multiplication.

```text
shape   = [2, 3]
strides = [3, 1]

index [1, 2]
offset = 1*3 + 2*1 = 5
```

Tensor v1 stores explicit contiguous strides even though they can be derived.
This lets invariants and future view semantics evolve without changing the
Tensor responsibility boundary.

For any zero-sized Shape, T1 uses canonical all-zero strides. No logical index
is valid for such a Shape, so these strides are never used to access storage;
the convention also avoids irrelevant stride-product overflow in empty layouts.

### Storage

Owns the native allocation as a concrete `Vec<f32>` with exclusive ownership.
Its API is intentionally limited to construction, zero allocation, length,
safe slice access, and transfer back into a Vec. Storage does not know logical
shape, strides, dtype, indexing semantics, devices, or mathematical operations.

### Tensor

Combines storage, shape, strides, and dtype and enforces their consistency:

- storage length equals Shape element count;
- strides have the same rank as Shape;
- strides describe contiguous row-major layout in v1;
- dtype is Float32;
- every reachable offset lies inside storage.

## Ownership and lifetime

Tensor v1 owns Storage directly. There is no `Arc`, reference counting, shared
mutable state, or borrowed view across the C ABI.

Future views may use shared immutable storage such as `Arc<Storage>`, but only
after creation, destruction, operation results, and PHP lifecycle behavior are
proven with exclusive ownership.

## Transpose policy

The existing buffer `transpose_f32` materializes reordered data and remains the
reference behavior. Tensor v1 should initially do the same and return a new
contiguous Tensor.

A future view can transpose without copying by swapping shape and strides:

```text
before: shape [2, 3], strides [3, 1]
after:  shape [3, 2], strides [1, 3]
```

That optimization is explicitly outside Tensor v1.

## Opaque C ABI boundary

The future ABI should expose an incomplete C type, never a Rust layout:

```c
typedef struct TransformerTensor TransformerTensor;
```

Lifecycle functions implemented in T5:

```c
int transformer_tensor_create_f32(
    const float* data,
    const size_t* shape,
    size_t rank,
    TransformerTensor** output
);

void transformer_tensor_destroy(TransformerTensor* tensor);
```

PHP treats the result as an opaque pointer. Successful creation writes one
uniquely owned handle to `output`; failure leaves it null and returns a status.
Exactly one matching destroy releases a non-null handle, while destroy of null
is a no-op. Double-destroy and use-after-free violate the caller contract. Rust
panics must be caught before they cross the ABI boundary.

Metadata queries use status codes and caller-owned output buffers. Shape is
copied rather than exposed through a borrowed pointer into Rust storage.

Future operations return new opaque handles:

```c
TransformerTensor* transformer_tensor_matmul(
    const TransformerTensor* a,
    const TransformerTensor* b
);
```

The concrete error-return mechanism must be designed before implementation;
returning null alone is insufficient for useful diagnostics.

## Materialization boundary

Tensor data remains native across operations. T6 provides
`transformer_tensor_numel` followed by `transformer_tensor_copy_data_f32` for
debugging, tests, serialization boundaries, and final PHP consumption. Copy-out
never writes when capacity is insufficient, never exposes internal storage, and
does not consume the handle.

## Reference APIs and parity

The current buffer APIs remain available during Tensor development:

```text
transformer_tensor_add_f32
transformer_matmul_f32
transformer_transpose_f32
transformer_softmax_f32
```

They are reference implementations for parity:

```text
validated buffer API ≈ Tensor-handle API
```

They must not be optimized away or deleted until Tensor operations have trusted
parity coverage and the project explicitly retires them.

## Implementation gates

Each gate requires its own tests and review before the next begins:

1. **T1 — DType, Shape, Strides:** complete in pure Rust; no FFI.
2. **T2 — Storage<Float32>:** complete with exclusive `Vec<f32>` ownership.
3. **T3 — Tensor Rust:** complete as a contiguous Float32 CPU Tensor; no handles.
4. **T4 — Opaque handle:** complete with a private Rust representation and lifecycle contract.
5. **T5 — Metadata ABI:** complete with create, destroy, shape, rank, and dtype.
6. **T6 — Explicit copy-out:** complete with numel query and capacity checks.
7. **T7 — Tensor add:** complete with buffer `add_f32` parity.
8. **T8 — Tensor matmul:** complete with naive buffer matmul parity.
9. **T9 — Tensor transpose:** complete as materialized and contiguous.
10. **T10 — Tensor softmax:** complete with stable 1D behavior and parity.

After T10, the project pauses Tensor kernel expansion for an NN architecture
review covering Parameter, Module, Linear, Embedding, LayerNorm, and GELU.
Views, shared storage, non-contiguous tensors, broadcasting, multiple dtypes,
and optimized kernels remain deferred until a concrete consumer requires them.

## Decisions validated in T5

Before implementation starts, confirm:

- status values are OK `0`, invalid argument `1`, panic `2`, and insufficient
  buffer `3`;
- shape queries copy into caller storage and return insufficient buffer when
  capacity is below rank;
- the stable Float32 dtype discriminant is `0`;
- create failures leave the output handle null;
- scalar and zero-sized Tensor creation follow the Rust Shape contracts;
- exactly-once PHP object destruction and formal ABI versioning remain for
  later integration work.
