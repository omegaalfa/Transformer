# Rust kernel benchmarks

This benchmark target measures the current scalar reference kernels without
changing their visibility or the runtime ABI. It uses only the Rust standard
library and always runs through Cargo's optimized `bench` profile.

Run the Transformer workload profile:

```bash
cargo bench --manifest-path runtime/Cargo.toml --bench kernels
```

Run the smaller smoke/CI profile:

```bash
TRANSFORMER_BENCH_PROFILE=quick \
    cargo bench --manifest-path runtime/Cargo.toml --bench kernels
```

Run the complete single-threaded matmul crossover sweep:

```bash
TRANSFORMER_BENCH_PROFILE=crossover \
TRANSFORMER_BENCH_FILTER=matmul \
TRANSFORMER_BENCH_SAMPLES=5 \
TRANSFORMER_BLAS_THREADS=1 \
    cargo bench --manifest-path runtime/Cargo.toml --bench kernels
```

The crossover profile measures `M = 1, 2, 4, 8, 16, 32, 64, 128, 256,
512` for each of `M x 768 * 768 x 768`, `M x 768 * 768 x 3072`, and
`M x 3072 * 3072 x 768`. Its output identifies the variant with the lowest
median latency as the winner for each shape. With five samples, p95 is the
slowest observed sample.

The following environment variables make runs configurable and reproducible:

- `TRANSFORMER_BENCH_PROFILE`: `transformer` (default), `quick`, or `crossover`;
- `TRANSFORMER_BENCH_FILTER`: `matmul`, `softmax`, `add`, or `transpose`;
- `TRANSFORMER_BENCH_SAMPLES`: timed sample count (default `15`, minimum `3`);
- `TRANSFORMER_BENCH_SAMPLE_MS`: target duration used to calibrate each sample
  (default `20`). A slow operation always runs at least once per sample.
- `TRANSFORMER_BLAS_THREADS`: OpenBLAS thread count (default `1`, matching the
  single-threaded Rust kernels).

The output is CSV-compatible and reports median and p95 latency. Matmul reports
the `current_scalar`, `cache_friendly`, `tiled`, `blas_sgemm`, and production
`dispatcher` variants, including speedup against both `current_scalar` and
`tiled` for the same shape.
The BLAS benchmark requires a system OpenBLAS installation exposing CBLAS and
links only the benchmark binary to `libopenblas`. Throughput is derived from
median latency:

- matmul: `2 * M * K * N` floating-point operations, reported as GFLOP/s;
- add: `12 * N` logical bytes (two Float32 reads and one write), as GB/s;
- transpose: `8 * N` logical bytes (one read and one write), as GB/s;
- softmax: `8 * N` logical input/output bytes, as GB/s. This intentionally does
  not count each internal pass, so implementations remain comparable.

For stable comparisons, use the same host, compiler/toolchain, power governor,
CPU affinity and background workload. Record `rustc -Vv` and the CPU model with
the results. Run each benchmark at least three separate times. The benchmark
does not include Tensor allocation, C ABI, PHP FFI, or host/device transfer.

The `current_scalar` baseline label identifies the unoptimized kernel
implementation present in `runtime/src/kernels/`.

The production dispatcher uses the benchmark-derived policy below. OpenBLAS is
resolved dynamically at runtime; when it is unavailable, calibrated matrix
shapes use the tiled Rust kernel instead:

- every `M = 1` matrix-vector operation uses `cache_friendly`;
- `768 -> 768` uses BLAS from `M = 4`;
- `768 -> 3072` and `3072 -> 768` use BLAS from `M = 2`;
- uncalibrated shapes retain `current_scalar` as the reference fallback.

## Layered runtime profiling

Build the release runtime, measure pure kernels, and then profile the native
buffer FFI, Tensor handle API, and complete PHP-array pipeline:

```bash
cargo build --release --manifest-path runtime/Cargo.toml
TRANSFORMER_BENCH_PROFILE=transformer \
    cargo bench --manifest-path runtime/Cargo.toml --bench kernels
TRANSFORMER_PROFILE_SAMPLES=5 php -d xdebug.mode=off \
    runtime/benches/profile_pipeline.php
```

The PHP profiler models one `128 x 768` Transformer-like activation pipeline:
one `768 -> 768` matmul, one residual add, softmax over the last dimension, and
one transpose. It explicitly compares 128 rank-1 softmax calls with one
rank-2 `[128, 768]` Tensor call. The pure-kernel benchmark likewise reports
`softmax_f32` for one row, 128 sequential rows, and
`softmax_last_dim_f32` for the complete contiguous matrix.

The profiler reports call counts, median/p95 latency, cost per call, and each
stage's percentage of its layer. Its `softmax_counts` lines count result
storage/handle allocations at the API contract level; they are not a global
allocator trace. C buffers are preallocated for the raw FFI layer. The legacy
PHP-array pipeline deliberately exercises array allocation and element-wise
copies, while the resident Tensor path performs three input copy-ins, no
intermediate copy-out, one batched softmax FFI call, and only one final
materialization. Disable Xdebug while measuring because its instrumentation
disproportionately affects PHP loops.

### Copy-out decomposition

The same profiler isolates materialization for 16, 98,304, 589,824 and
1,048,576 elements. For each size it reports:

- `metadata_numel`: fixed Tensor metadata call;
- `allocate_cdata`: temporary contiguous `float[N]` allocation;
- `ffi_copy_preallocated`: Rust storage to an already allocated CData buffer;
- `cdata_to_php_array_current`: current loop with generic numeric validation;
- `single_guard_candidate`: experimental loop with one exact `float` guard;
- `string_unpack_candidate`: rejected `FFI::string()`/`unpack()` alternative;
- `full_current`/`full_single_guard_candidate`: complete controlled paths with
  identical setup, differing only in dynamic validation;
- `to_float32_current`: complete public operation.
- `export_float32_buffer`: one Rust-to-CData copy without float zvals;
- `buffer_value_at`: one-value consumer;
- `buffer_to_bytes`: binary little-endian export;
- `buffer_hash_consumer`: complete buffer-to-binary-consumer path;
- `buffer_destroy`: explicit release cost.

The `FFI::string()`/`unpack()` candidate remains benchmark-only because it was
slower. The single-guard path is used in production: alternating A/B samples on
the target `[128, 768]` buffer showed a material conversion speedup while
preserving the existing ABI and result. Allocation counts describe observable
CData/PHP result allocations, not internal Zend allocator calls.

`pipeline_totals_us` reports `native_without_export`, `native_with_array`,
`native_with_buffer`, and `native_buffer_and_hash`, so the structural cost of
returning a normal PHP array remains visible.

## Rejected `matmul + add` fusion experiment

Run the isolated A/B experiment with:

```bash
TRANSFORMER_BENCH_FILTER=fusion \
TRANSFORMER_BENCH_SAMPLES=15 \
TRANSFORMER_BENCH_SAMPLE_MS=10 \
TRANSFORMER_BLAS_THREADS=1 \
    cargo bench --manifest-path runtime/Cargo.toml --bench kernels
```

The baseline allocates an intermediate matmul buffer and a separate add output.
The candidate lets the existing dispatcher (including OpenBLAS where selected)
write into the output and then adds the residual in place. Inputs remain
unchanged. Each shape is checked bit-for-bit before timing. Samples are paired,
use the same calibrated iteration count, and alternate A/B execution order.

The handle, buffer, and FFI columns printed by this pure-kernel benchmark model
what a public fused operation would require: `2 -> 1` result handles, `2 -> 1`
result buffers, and `2 -> 1` calls. No public fused operation was added.

Measured on OpenBLAS 0.3.26 (`pthreads`, one thread, Haswell); times are
median/p95 in microseconds:

| M | 768 -> 768 baseline | candidate | speedup |
|---:|---:|---:|---:|
| 1 | 111.978 / 300.688 | 103.328 / 384.418 | 1.084x |
| 2 | 315.994 / 494.700 | 311.954 / 481.598 | 1.013x |
| 4 | 217.178 / 367.030 | 220.464 / 351.950 | 0.985x |
| 8 | 245.144 / 303.821 | 245.328 / 453.372 | 0.999x |
| 16 | 473.832 / 666.342 | 483.501 / 1184.438 | 0.980x |
| 32 | 728.449 / 1215.081 | 670.287 / 934.080 | 1.087x |
| 64 | 1002.669 / 1089.196 | 1023.497 / 1233.446 | 0.980x |
| 128 | 1904.307 / 2943.932 | 1848.565 / 5715.704 | 1.030x |

| M | 768 -> 3072 baseline | candidate | speedup |
|---:|---:|---:|---:|
| 1 | 815.694 / 1231.772 | 845.165 / 1243.527 | 0.965x |
| 2 | 1276.948 / 2561.597 | 1311.064 / 2431.590 | 0.974x |
| 4 | 1085.256 / 1217.140 | 1093.388 / 1236.323 | 0.993x |
| 8 | 1303.520 / 1961.329 | 1319.942 / 1892.351 | 0.988x |
| 16 | 1788.336 / 2953.920 | 1818.440 / 2759.084 | 0.983x |
| 32 | 2653.972 / 3851.244 | 2625.706 / 4239.170 | 1.011x |
| 64 | 4535.849 / 6166.010 | 4659.677 / 5194.369 | 0.973x |
| 128 | 8157.504 / 9018.919 | 8109.293 / 10724.798 | 1.006x |

| M | 3072 -> 768 baseline | candidate | speedup |
|---:|---:|---:|---:|
| 1 | 737.439 / 826.351 | 709.933 / 777.937 | 1.039x |
| 2 | 1185.764 / 1537.037 | 1181.501 / 1567.524 | 1.004x |
| 4 | 1048.250 / 1203.294 | 1058.400 / 1714.799 | 0.990x |
| 8 | 1264.755 / 1407.880 | 1276.161 / 1364.782 | 0.991x |
| 16 | 1628.063 / 2366.851 | 1605.514 / 1741.725 | 1.014x |
| 32 | 2379.369 / 2472.595 | 2362.622 / 2511.926 | 1.007x |
| 64 | 3821.556 / 4833.441 | 3800.039 / 4333.766 | 1.006x |
| 128 | 6865.332 / 8306.200 | 6745.970 / 7536.539 | 1.018x |

The candidate fails the acceptance gate: the best median improvement is 8.7%,
several shapes regress, and p95 is not consistently better. The measured
resident `[128, 768]` pipeline baseline is 2036.368 us without export; even the
55.742 us kernel saving at that shape could improve the complete pipeline by
only about 2.7%. Therefore the fusion was reverted from production and remains
benchmark-only. Preserving the current two-operation API is preferable to the
extra kernel/FFI/lifecycle surface for this result.

## Rejected owned-storage reuse experiment

The resident Tensor ABI preserves every input. Consequently, an operation sees
only immutable pointers and cannot determine whether PHP has another reference
to a `NativeStorage`. Reusing an input buffer behind this contract would create
observable mutation and possible use-after-free. Reference counting in PHP does
not provide Rust with a sound exclusivity proof.

The allocation path for every result is:

```text
Vec<f32> -> Storage -> Tensor -> Box<TransformerTensor> -> opaque PHP handle
```

Destroying the opaque handle reconstructs the `Box`; dropping it recursively
drops Tensor, Storage and Vec. A null destroy is accepted, but a non-null native
handle must still be destroyed exactly once. PHP makes its wrapper-level
`destroy()` idempotent by clearing the CData handle.

For the current `[128, 768]` profiler pipeline, structural counts are:

| Operation | Vec | Tensor | Box/handle | Destroy | Output bytes |
|---|---:|---:|---:|---:|---:|
| matmul | 1 | 1 | 1 | 1 | 393,216 |
| add | 1 | 1 | 1 | 1 | 393,216 |
| softmax_last_dim | 1 | 1 | 1 | 1 | 393,216 |
| transpose | 1 | 1 | 1 | 1 | 393,216 |
| four outputs | 4 | 4 | 4 | 4 | 1,572,864 |
| three inputs plus outputs | 7 | 7 | 7 | 7 | 4,718,592 |

These are code-derived lifecycle counts, not global allocator events. Shape and
stride vectors may have their own small allocations and allocator capacity is
not observable without global instrumentation. The profiler measured seven
native handle destructions at 6.592 us median through the Tensor ABI and 15.112
us through PHP. The complete resident path without export took 2038.419 us.

### Experimental upper bound

The benchmark-only candidate models an explicitly consuming internal chain. It
owns one boxed temporary plus a scratch Vec and swaps their buffers after each
unchanged kernel. Rust ownership guarantees that the consumed temporary cannot
be accessed again. This is deliberately not connected to production, FFI or
PHP.

Run it with:

```bash
TRANSFORMER_BENCH_FILTER=lifecycle \
TRANSFORMER_BENCH_SAMPLES=15 \
TRANSFORMER_BENCH_SAMPLE_MS=20 \
TRANSFORMER_BLAS_THREADS=1 \
    cargo bench --manifest-path runtime/Cargo.toml --bench kernels
```

The benchmark checks bitwise parity at every shape before timing. A separate
preallocated-kernel measurement is printed, but subtracting independent
medians is only diagnostic: host variance can make the resulting lifecycle
estimate zero after saturation. Total paired A/B latency is the decision
metric. Results below are median/p95 microseconds from the repeated run:

| Shape | Baseline | Reuse | Speedup |
|---|---:|---:|---:|
| 1x768 -> 768 | 124.695 / 177.432 | 119.968 / 197.180 | 1.039x |
| 2x768 -> 768 | 231.437 / 267.859 | 224.468 / 274.258 | 1.031x |
| 4x768 -> 768 | 220.010 / 274.203 | 212.360 / 257.391 | 1.036x |
| 8x768 -> 768 | 314.392 / 364.572 | 304.869 / 354.953 | 1.031x |
| 16x768 -> 768 | 462.508 / 674.692 | 460.854 / 808.920 | 1.004x |
| 32x768 -> 768 | 969.177 / 2233.090 | 1054.669 / 1781.171 | 0.919x |
| 64x768 -> 768 | 1702.521 / 2913.971 | 1616.421 / 3220.959 | 1.053x |
| 128x768 -> 768 | 3294.662 / 7119.314 | 3601.186 / 8275.991 | 0.915x |
| 1x768 -> 3072 | 877.423 / 2238.354 | 871.163 / 3468.934 | 1.007x |
| 2x768 -> 3072 | 1527.193 / 1841.151 | 1417.687 / 2772.415 | 1.077x |
| 4x768 -> 3072 | 1285.038 / 1551.574 | 1227.140 / 1533.849 | 1.047x |
| 8x768 -> 3072 | 1509.973 / 2079.139 | 1575.494 / 2157.012 | 0.958x |
| 16x768 -> 3072 | 2323.632 / 2971.418 | 2269.086 / 2794.874 | 1.024x |
| 32x768 -> 3072 | 3784.540 / 4666.629 | 3802.875 / 7409.602 | 0.995x |
| 64x768 -> 3072 | 6345.562 / 7136.699 | 6571.031 / 8400.309 | 0.966x |
| 128x768 -> 3072 | 12039.155 / 13629.273 | 12117.343 / 15122.644 | 0.994x |
| 1x3072 -> 768 | 898.506 / 1076.601 | 867.117 / 1003.559 | 1.036x |
| 2x3072 -> 768 | 1241.791 / 2054.749 | 1229.628 / 1899.990 | 1.010x |
| 4x3072 -> 768 | 1172.644 / 1982.231 | 1146.460 / 1945.345 | 1.023x |
| 8x3072 -> 768 | 1465.189 / 2126.516 | 1429.015 / 1643.500 | 1.025x |
| 16x3072 -> 768 | 1807.519 / 2539.292 | 1841.873 / 2030.924 | 0.981x |
| 32x3072 -> 768 | 2791.447 / 3234.442 | 2761.689 / 3193.717 | 1.011x |
| 64x3072 -> 768 | 4441.908 / 5429.217 | 4490.182 / 4801.221 | 0.989x |
| 128x3072 -> 768 | 8093.321 / 9579.811 | 8241.082 / 9439.300 | 0.982x |

| Metric | Baseline | Owned reuse | Delta |
|---|---:|---:|---:|
| accumulated output Vec allocations | 4 | 2 | -50% |
| accumulated output allocation bytes | `4 * M * N * 4` | `2 * M * N * 4` | -50% |
| modeled result handles created | 4 | 1 | -75% |
| peak live result handles | 2 | 1 | -50% |
| minimum peak output bytes | `2 * M * N * 4` | `2 * M * N * 4` | unchanged |

Storage reuse is rejected. Median gains are inconsistent and stay below 10%,
several shapes regress, p95 often worsens, and prompt destruction already gives
the baseline the same minimum peak buffer memory. Making the experiment usable
from PHP would require a consuming ABI and change current semantics.

### Architectural next step (not implemented)

A future graph executor could safely recycle storage because it would own an
operation DAG rather than infer liveness from opaque FFI pointers. During graph
planning it should calculate use counts and last-use positions, assign
compatible row-major Float32 buffers from a size-class pool, and release a
buffer only after its final consumer. External outputs and aliased graph inputs
must be pinned. Execution errors must return all exclusively owned temporaries
to the pool without invalidating externally visible tensors. This design can
also decide when an intermediate truly requires materialization. The following
simulation implements only this planning proof; it does not introduce a
consuming ABI or public API.

## Experimental graph lifetime planner

The internal, simulation-only DAG planner is compiled into the Rust crate for
tests but is not called by production, FFI or PHP. It separates:

```text
logical Value (DAG identity and metadata)
    != external native handle (caller ownership)
    != physical StorageSlot (recyclable capacity)
```

Supported node descriptions are limited to `Matmul`, `Add`,
`SoftmaxLastDim`, and `Transpose`; no mathematical kernel is executed or
changed. Nodes must be added in topological order. The explicit lifetime pass
computes consumer lists, counts and the last consumer of every value.

For each produced value, the first planner uses the closed interval
`[producer_node, last_consumer_node]`. An external output extends through the
end of the graph. Two values may share a slot only when their intervals do not
overlap (`end(A) < start(B)`), dtype and layout match, and existing capacity is
large enough. External inputs remain caller-owned and receive no pool slot.

The experimental `StoragePool` simulator allocates planned slots lazily,
records reuse and releases, and pins external outputs. It contains no `Vec`,
Tensor, native pointer or allocator integration; its metrics describe the
proven plan rather than production allocation events.

Run the planner benchmark:

```bash
TRANSFORMER_GRAPH_BENCH_SAMPLES=25 \
TRANSFORMER_GRAPH_BENCH_TARGET_US=1000 \
    cargo bench --manifest-path runtime/Cargo.toml --bench graph
```

Measured release results on this host, in median/p95 microseconds:

| DAG | Nodes | Construction | Lifetime | Storage plan | Simulation | Slots | Allocations | Reuses | Peak bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Transformer | 4 | 0.670 / 1.460 | 0.031 / 0.040 | 0.156 / 0.162 | 0.642 / 0.686 | 2 | 2 | 2 | 786,432 |
| Linear | 10 | 1.517 / 3.365 | 0.049 / 0.067 | 0.292 / 0.337 | 1.941 / 2.228 | 2 | 2 | 8 | 786,432 |
| Branching | 5 | 0.760 / 1.209 | 0.037 / 0.119 | 0.179 / 0.202 | 0.847 / 0.905 | 3 | 3 | 2 | 1,179,648 |
| Linear | 100 | 13.194 / 22.537 | 0.478 / 0.528 | 6.297 / 7.565 | 14.521 / 19.730 | 2 | 2 | 98 | 786,432 |
| Linear | 1000 | 175.235 / 196.057 | 9.194 / 40.037 | 581.017 / 858.377 | 144.272 / 162.709 | 2 | 2 | 998 | 786,432 |

For the exact `[128,768]` Transformer chain, the plan assigns matmul output and
softmax output to slot 0, then add output and the external transpose output to
slot 1. Compared with four retained output buffers, this proves a theoretical
reduction from four physical allocations (1,572,864 bytes) to two allocations
(786,432 bytes), with two reuse events and a 50% lower planned peak.

The branch test keeps the matmul value alive through both Softmax and Transpose
consumers. It therefore needs three slots and cannot recycle the matmul slot at
the first branch. Tests also pin external intermediate/output values, reject
incompatible shapes/dtypes/layouts and capacities, and reject planning before
lifetime analysis.

This is a linear-order interval approximation, not a general graph-coloring or
scheduling framework. The greedy slot search checks previously assigned values
and is quadratic in the worst case; this is visible in the 1000-node storage
plan. Even there, planning costs about 0.58 us per scheduled node, far below the
hundreds of microseconds to milliseconds measured for the real Transformer
kernels. No runtime speedup is claimed because the executor remains a
simulator and performs string-based event tracing.

## Destination-storage integration audit

The graph planner remains disconnected from numerical execution. This audit
defines the smallest future bridge between a `StoragePlan` and the existing
kernels; it does not implement that bridge.

### Existing kernel contracts

Output allocation is not performed by any kernel below. The resident Tensor
FFI currently creates `vec![0.0; output_numel]`, calls the kernel with its
mutable slice, then wraps that Vec in `Storage`, `Tensor` and an opaque Box.

| Operation | Current destination contract | Shape known before execution | Initialization | Destination integration |
|---|---|---|---|---|
| `matmul_f32` | exact `M*N` mutable slice | yes, `M x N` | no prior value required; every cell is assigned, including zero for `K=0` | already accepts caller-provided output |
| `matmul_dispatch_f32` | forwards the same exact mutable slice | yes, `M x N` | backend-dependent internally | already accepts caller-provided output without changing policy |
| `add_f32` | exact slice matching both inputs | yes, same shape as inputs | no; every element is assigned | already accepts caller-provided output |
| `softmax_f32` | exact non-empty slice matching input | yes, same shape | no; exponentials are written before normalization | already accepts caller-provided output |
| `softmax_last_dim_f32` | exact contiguous slice, divisible by non-zero last dimension | yes, same shape | no; delegates every row to `softmax_f32` | already accepts caller-provided output |
| `transpose_f32` | exact `rows*columns` mutable slice | yes, `columns x rows` | no; every destination index is assigned | already accepts caller-provided output |

All kernels validate exact lengths. A destination bridge must additionally
validate dtype Float32, contiguous row-major layout, logical output shape,
capacity in elements, checked byte size, and exclusive mutable ownership for
the duration of the call. A pooled Vec remains initialized Rust memory; using
uninitialized allocation would require a separate unsafe proof and is outside
this design.

### Compatibility and aliasing matrix

| Operation | External destination | Exact input/output alias with current algorithm | Shape known | Planned storage reuse |
|---|---|---|---|---|
| Matmul | safe today as a distinct exact slice | unsafe with A or B; Rust kernels overwrite data needed later and cache/tiled paths first zero C; SGEMM also requires distinct C | yes | safe only from a logically dead, non-operand value |
| Add | safe today as a distinct exact slice | algorithmically safe with exactly A or exactly B, but impossible through the current safe Rust signature; partial overlap is unsafe | yes | safe with a distinct dead slot; in-place would require a separate implementation/signature |
| SoftmaxLastDim | safe today as a distinct exact slice | exact alias is algorithmically possible only because maximum validation finishes before the write pass and each source element is then read once; current Rust borrows still forbid it | yes | safe with a distinct dead slot; in-place requires an explicit tested kernel |
| Transpose | safe today as a distinct exact slice | unsafe except accidental degenerate shapes; writes can destroy source elements not yet read | yes, reversed rank-2 shape | safe only with a distinct dead slot |

The planner must never turn storage-slot equality into implicit input/output
aliasing. Closed lifetimes prevent an output from occupying an input's slot at
the same node. Reuse means that a previous logical value ended at an earlier
node; it does not mean that a kernel reads and writes the same live value.

#### Concrete alias cases

- `Add(X, R) -> X`: mathematically and algorithmically valid for exact alias
  because index `i` is read before index `i` is written and no later iteration
  reads it. The current `&[f32]` plus `&mut [f32]` API cannot legally receive
  aliased slices, so a future executor must still use a distinct slot unless a
  dedicated in-place kernel is introduced.
- `Softmax(X) -> X`: the maximum/non-finite pass completes before writes. In
  the exponential pass, each input is consumed before its corresponding output
  is stored; normalization reads only output. Exact alias is therefore possible
  for the present algorithm, including row-by-row last-dimension softmax, but
  must not be inferred from this fact or constructed with aliased Rust slices.
- `Transpose(X) -> X`: unsafe for the current nested loops. For example, a
  write to a transposed destination can replace an element needed by a later
  source index. Rectangular matrices also require cycle handling for a genuine
  in-place algorithm.
- `Matmul(A, B) -> A` or `-> B`: unsafe. Scalar traversal can overwrite future
  operands; cache-friendly and tiled implementations call `output.fill(0.0)`;
  BLAS C must not overlap A or B. Output shape may also differ from either
  operand.

### OpenBLAS and dispatcher

The dispatcher already receives the executor-compatible shape:
`(A, B, &mut C, M, K, N)`. It can continue selecting reference,
cache-friendly, tiled or BLAS without knowing whether C came from a fresh Vec
or a planned slot.

The OpenBLAS path calls `cblas_sgemm` with:

```text
layout = CblasRowMajor
transA = CblasNoTrans
transB = CblasNoTrans
M, N, K
alpha = 1
lda = K
ldb = N
beta = 0
ldc = N
C = destination M x N
```

Thus OpenBLAS already writes directly into destination storage with no layout
conversion or copy. The bridge must retain the existing checked `M*K`, `K*N`
and `M*N` lengths, C `int` dimension conversion, contiguous row-major layout,
and non-overlap of C with A/B. `beta=0` makes previous slot contents irrelevant.
For a zero dimension the wrapper fills C with zero. If BLAS is unavailable, the
same destination passes unchanged to the tiled fallback; dispatcher policy does
not need modification.

### Physical transpose versus a view

The current physical transpose is necessary whenever a downstream consumer
expects the transposed contiguous row-major order. Although `Tensor` stores
strides, its constructor always creates contiguous strides and it exclusively
owns a Storage whose length exactly matches its shape. There is no offset,
shared-storage ownership, or view lifetime model.

A future transpose view could represent reversed shape/strides without copying,
but the current kernels cannot consume it:

- matmul and OpenBLAS assume contiguous row-major operands and fixed leading
  dimensions;
- add pairs equal linear offsets;
- last-dimension softmax uses contiguous chunks;
- transpose itself indexes a contiguous source.

Introducing views would therefore require storage sharing with explicit alias
and lifetime rules, offsets, arbitrary strides in kernel contracts, and likely
materialization barriers before BLAS or unsupported kernels. Eliminating a
transpose is potentially more valuable than tiling it because it removes all
reads/writes and the output buffer, but only graph-wide layout propagation can
prove that elimination. A tiled transpose may improve cache locality if a
physical materialization remains necessary; neither change is implemented or
credited with a speedup here.

### Proposed internal `DestinationStorage`

A future internal bridge can remain a thin ownership guard over the existing
Storage rather than a second Tensor abstraction:

```text
DestinationStorage
  dtype: DType
  layout: Layout
  logical_shape: Shape
  capacity_elements: usize
  slot: StorageSlotId
  state: Available | Borrowed(NodeId) | PinnedExternal(ValueId)
  storage: Storage
```

Conceptual operations:

```text
acquire(slot, expected_metadata) -> DestinationLease
DestinationLease::as_exact_f32_slice_mut(numel) -> &mut [f32]
commit(value_id) -> planned value binding
release(value_id) -> pool availability after proven last_use
```

The lease supplies the exclusive mutable borrow. Inputs are borrowed separately
from different live slots or external Tensors. Acquisition must fail before any
kernel call if dtype, layout, capacity, ownership state or binding differs from
the plan. On kernel error, the lease returns to its prior available state and no
logical output is published. External outputs transition to a pinned state.

### Proposed `ExecutionPlan`

```text
ExecutionPlan
  graph: finalized logical graph
  storage_plan: StoragePlan
  nodes: Vec<NodeExecution>

NodeExecution
  node: NodeId
  operation: Matmul | Add | SoftmaxLastDim | Transpose
  inputs: Vec<InputBinding>
  output: OutputBinding

InputBinding = External(ValueId) | Slot(ValueId, StorageSlotId)
OutputBinding = Slot(ValueId, StorageSlotId) | ExternalOutput(ValueId, StorageSlotId)
```

For the Transformer pipeline:

```text
Matmul:  A external, B external       -> X slot 0
Add:     X slot 0, residual external  -> Y slot 1; release X after node
Softmax: Y slot 1                     -> Z slot 0; release Y after node
Transpose: Z slot 0                   -> O slot 1 pinned; release Z after node
```

Execution must acquire the output lease before invoking the unchanged kernel,
commit only on success, decrement logical use counts, and release inputs only
at their planner-proven last use. The runtime must validate that execution node
order and bindings match the finalized graph rather than accepting arbitrary
slot IDs from callers.

### Future integration cost

Internal changes that can preserve all public contracts:

- add `DestinationStorage`, leases and real pooled Storage behind the graph
  module;
- add `ExecutionPlan`/`NodeExecution` generated from finalized Graph and
  StoragePlan;
- add internal Tensor construction from a committed pooled Storage;
- call the existing kernels and dispatcher with exact destination slices;
- add rollback/pinning logic and numerical A/B tests against ordinary Tensor
  operations.

No existing kernel signature needs to change for distinct destinations. The
Tensor FFI and PHP API can remain unchanged if graph execution remains internal
and returns an ordinary owned Tensor/handle only for external outputs. Actually
submitting graphs or execution plans from PHP would require new additive FFI
and PHP APIs; replacing or changing existing symbols is unnecessary and is not
proposed.

Required future tests include plan/binding mismatch, lease double-acquire,
release before last use, external pinning, rollback after every kernel error,
capacity/dtype/layout mismatch, BLAS and every Rust fallback, zero dimensions,
branching, simultaneous outputs, output delivery ownership, and numerical
parity for all established Transformer shapes.

### Permanent optimization matrix

| Area | Current measured state | Investigated/rejected | Potential future work | Decision basis |
|---|---|---|---|---|
| Matmul | dispatcher; cache-friendly M=1; optional single-thread OpenBLAS at calibrated thresholds | scalar reordering/tiling measured; matmul+add fusion rejected | executor-provided distinct C; recalibrate per target/backend | BLAS dominates calibrated shapes; fusion did not reach 10% pipeline gain |
| Add | separate linear kernel; about 54-73 us for 98,304 elements in measured runs | matmul+add fusion and naive local reuse rejected | distinct planned destination; explicit in-place variant only with evidence | already a small pipeline fraction; local changes were noisy/inconsistent |
| Softmax rank-1 | stable scalar kernel; 128-call FFI pattern measured | repeated rank-1 pipeline superseded for resident matrices | retain for vectors; optimize only from new profiling | per-call path is correct but call count dominated matrices |
| Softmax last-dim | one batched rank-N call; roughly 0.74-0.79 ms at `[128,768]` | 128 FFI calls reduced to one | planned destination; explicit in-place or SIMD only after benchmark | batching removed lifecycle/FFI repetition; still a measured kernel hotspot |
| Transpose | physical contiguous copy; roughly 0.35-0.38 ms at `[128,768]` | no tiled/SIMD implementation yet; naive alias reuse is unsafe | graph-wide view/layout propagation first; tiled physical fallback second | eliminating materialization is structurally stronger than only accelerating it |
| Materialization | PHP array conversion about 4.4-5.9 ms for 98,304 elements | `FFI::string`/`unpack` rejected; redundant validation removed | keep arrays for compatibility; binary consumers use Float32Buffer | PHP zval creation dominates; Float32Buffer avoids it when applicable |
| Storage allocation | four result Vec/Tensor/Box/handles in current resident profile | naive owned reuse rejected: inconsistent and below 10% | planned slots with leases and rollback | DAG proves two slots and 50% theoretical output memory reduction |
| Graph execution | deterministic lifetime/slot simulator; 0.187 us lifetime+planning for four nodes | local heuristics and refcount inference rejected | real internal DestinationStorage/ExecutionPlan experiment | planner proves liveness and branching but no runtime speedup is claimed |

This matrix is the project record for previously evaluated optimizations. New
work should update measured values and decisions instead of reopening rejected
approaches without new evidence.
