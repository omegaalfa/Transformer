# Native Rust runtime

The native runtime is a standalone Rust crate under `runtime/`, built by Cargo
as a `cdylib`. It is not implemented inside the PHP extension.

```text
PHP library -> FFI -----------------+
                                      -> stable C ABI -> Rust runtime
PHP library -> Zend extension ------+
```

PHP never receives Rust strings, vectors, traits, panics, or ownership types.
The C ABI uses pointers, lengths, primitive values, and integer status codes.

## Safety boundary

Unsafe code is confined to `runtime/src/ffi/mod.rs`, where validated pointers
are converted into borrowed slices. Kernels such as
`runtime/src/kernels/add.rs` are safe Rust and accept only slices. Exported
operations use `catch_unwind` so a panic cannot unwind through the C ABI.

`catch_unwind` contains Rust panics; it cannot make an invalid pointer safe.
For every non-empty buffer call, the ABI caller guarantees that:

- `a` points to at least `length` initialized, readable `f32` elements;
- `b` points to at least `length` initialized, readable `f32` elements;
- `output` points to at least `length` writable `f32` elements;
- all three buffers remain valid for the complete call;
- `output` does not overlap the input buffers.

The boundary can reject null pointers and impossible lengths, but a non-null
pointer is not proof that memory is valid. Violating the caller contract may
cause undefined behavior before Rust can return a status code.

The addition ABI returns:

```text
0 = OK
1 = INVALID_ARGUMENT
2 = PANIC contained at the ABI boundary
```

## Opaque Tensor decision

```text
PHP Tensor
    |
    +-- handle / native pointer
              |
              v
         Native Tensor
              +-- buffer
              +-- shape
              +-- strides
              +-- dtype
              +-- device
```

Native buffers will remain native between operations. Conversion to PHP arrays
will be explicit. Rust ownership and borrowing will govern native lifetimes;
the C ABI will expose opaque handles rather than Rust types.

## Current copying model

The experimental API intentionally performs copies:

```text
PHP array -> FFI buffer -> Rust kernel -> FFI buffer -> PHP array
```

This is appropriate for validating buffer layout, ABI calls, status handling,
and numerical parity. It is not the final Transformer execution model because
copying through PHP between every operation would dominate runtime costs.

Only after the basic reference kernels are stable will PHP Tensor objects hold
opaque handles to Rust-owned tensors. Operations will then keep buffers in Rust
memory, and conversion back to PHP arrays will be explicit.

The intended progression is:

```text
native_version -> add_f32 -> matmul_f32 -> transpose_f32 -> softmax_f32
    -> stable basic operations
    -> Native Tensor / Shape / Strides / DType / opaque handles
```

Future kernels include normalization, attention, activation, cache, and
quantization. SIMD, BLAS, threads, GPU, and optimized matmul follow correctness,
parity, and profiling. The naive matmul stays available as the numerical
reference rather than being replaced by optimized implementations.

The 1D softmax reference is numerically stable: it subtracts the maximum input
before exponentiation, then normalizes by the shifted exponential sum. Axis,
batch, and multidimensional softmax wait for the Tensor/Shape design.
