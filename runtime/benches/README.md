# Rust kernel benchmarks

## GPU-R1 BGE end-to-end benchmark

Build the opt-in CUDA runtime and run the real-checkpoint benchmark:

```bash
cargo build --release --manifest-path runtime/Cargo.toml --features cuda
TRANSFORMER_BGE_CUDA_CHECKPOINT=/path/to/bge-small-en-v1.5 \
TRANSFORMER_BGE_CUDA_SAMPLES=15 \
TRANSFORMER_BGE_CUDA_SOAK=100 \
php -d xdebug.mode=off runtime/benches/bge_cuda.php
```

The runner includes five warmups, official Float32 parity, 15 latency samples,
parameter-handle identity, determinism, and CUDA free-memory checks across 100
additional forwards. On the RTX 3060 validation host it measured p50/p95
`8.675/12.759 ms`; maximum absolute/relative error was
`1.0430813e-7/3.5754e-5`, all 197 parameters remained resident, and free-memory
delta after the soak was zero. The only forward transfers are IDs/mask/types
H2D and the final `[B,384]` embedding D2H.

GPU-R2 adds CUDA Event profiling with:

```bash
TRANSFORMER_BGE_CUDA_CHECKPOINT=/path/to/bge-small-en-v1.5 \
TRANSFORMER_BGE_CUDA_PROFILE_SAMPLES=5 \
php -d xdebug.mode=off runtime/benches/bge_cuda_profile_events.php
```

The optimized path uses a persistent workspace/stream and parallel LayerNorm,
attention, and CLS/L2 kernels. The final 25-sample short-input run measured
p50/p95 `3.260/10.195 ms`, compared with GPU-R1 `8.675/12.759 ms`. Steady-state
allocation/free counts are zero, the logical launch count is 207, and one
explicit stream synchronization occurs at final D2H. TF32 and cuBLAS batched
attention were measured and rejected; strict FP32 and specialized attention
remain production defaults.

## MODEL-R5 BGE end-to-end benchmark

`bge_embedding.php` measures the public sentence-to-embedding path in one PHP
process. It loads config, tokenizer, Safetensors and all 197 Parameters once,
then measures warm `encode()`/`encodeBatch()`, recreate versus resident use and
RSS. The public methods include tokenization, BERT, official CLS pooling, L2
normalization and final PHP-list materialization.

```bash
TRANSFORMER_BGE_CHECKPOINT=/tmp/transformer-model-r3/bge-small-en-v1.5 \
TRANSFORMER_BGE_SAMPLES=15 \
TRANSFORMER_BGE_RECREATE_SAMPLES=3 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
php -d xdebug.mode=off runtime/benches/bge_embedding.php
```

The controlled reference run used one OpenBLAS thread. Warm single used 15
samples after three warmups. The complete batch matrix used three samples per
cell because the B=32/66-token case alone takes roughly 82 seconds per sample;
therefore its p95/p99 equal the observed maximum and are descriptive rather
than robust tail estimates. Effective lengths were short=9, medium=26 and
long=66 WordPiece tokens.

Cold load took 10,226.725 ms. Warm short single encode measured p50/p95/p99
347.567/381.050/381.050 ms, mean 352.514 ms, min 338.052 ms and max 381.050 ms.

| Tokens | Batch | p50 ms | p95/p99 ms | Mean sentences/s |
|---:|---:|---:|---:|---:|
| 9 | 1 | 345.181 | 351.627 | 2.884 |
| 9 | 2 | 694.533 | 707.195 | 2.886 |
| 9 | 8 | 2,834.398 | 3,681.089 | 2.583 |
| 9 | 16 | 5,693.875 | 6,040.748 | 2.826 |
| 9 | 32 | 11,422.180 | 11,690.759 | 2.813 |
| 26 | 1 | 1,001.904 | 1,016.230 | 0.997 |
| 26 | 2 | 2,251.663 | 2,616.169 | 0.876 |
| 26 | 8 | 8,827.401 | 9,028.633 | 0.900 |
| 26 | 16 | 16,080.239 | 17,538.863 | 0.972 |
| 26 | 32 | 32,446.991 | 35,203.270 | 0.964 |
| 66 | 1 | 2,732.832 | 2,869.942 | 0.368 |
| 66 | 2 | 5,570.043 | 9,547.648 | 0.296 |
| 66 | 8 | 20,398.461 | 21,259.621 | 0.388 |
| 66 | 16 | 40,965.032 | 41,450.858 | 0.390 |
| 66 | 32 | 82,163.675 | 84,247.000 | 0.391 |

For a 9-token sentence, resident mean latency was 359.598 ms versus
10,055.990 ms for load+encode+destroy, a 27.965x difference. Parameter storage
identities remained unchanged. RSS was 285,560,832 bytes after load,
287,346,688 after warmup and 375,697,408 after the matrix; ten subsequent
forwards remained exactly at 375,697,408 bytes, showing no progressive growth
in this workload. Peak PHP-reported allocation was 728,469,504 bytes.

The pre-benchmark official reference check passed with maximum absolute error
1.9371509552001953e-7 and maximum relative error
5.0901316382054526e-5 under the unchanged MODEL-R4 Float32 tolerance.

## MODEL-R6 BGE dispatcher calibration

Profiling showed that every BGE projection (`384→384`, `384→1536` and
`1536→384`) was classified as unknown and sent to `current_scalar`. The
production dispatcher now uses the measured BGE policy: M=1 remains
cache-friendly, M=2 uses the tiled fallback, and OpenBLAS is selected from M=4.
The historical 768/3072 rules are unchanged.

Reproduce kernel calibration and the real-model stage profile with:

```bash
TRANSFORMER_BENCH_PROFILE=crossover \
TRANSFORMER_BENCH_FILTER=matmul \
TRANSFORMER_BENCH_PROJECTION=384x1536 \
TRANSFORMER_BENCH_MAX_M=72 \
TRANSFORMER_BENCH_SAMPLES=3 \
TRANSFORMER_BLAS_THREADS=1 \
    cargo bench --manifest-path runtime/Cargo.toml --bench kernels

TRANSFORMER_BENCH_FILTER=bert_attention_profile \
TRANSFORMER_BENCH_SAMPLES=15 \
TRANSFORMER_BLAS_THREADS=1 \
    cargo bench --manifest-path runtime/Cargo.toml --bench kernels

TRANSFORMER_BGE_PROFILE_SAMPLES=5 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    php -d xdebug.mode=off runtime/benches/bge_forward_profile.php
```

For B=1/S=9, the scalar profile attributed 28.73% to attention, 32.92% to
FFN input Linear and 34.42% to FFN output Linear. After calibrated dispatch,
attention was 34.64%, both FFN Linears together were 29.00%, ExactGELU was
28.38%, residual/LayerNorm was 5.38%, and embeddings/pooling/L2/final PHP
materialization together were 1.28%. The dominant bottleneck therefore moved
from scalar GEMM to native attention and ExactGELU.

The 25-sample optimized short encode measured p50/p95/p99
34.682/41.169/42.863 ms versus the MODEL-R5 p50/p95 of
347.567/381.050 ms: 10.02x median speedup and a 89.2% lower p95. The complete
15-sample matrix measured:

| Tokens | Batch | p50 before | p50 after | Speedup | After sentences/s |
|---:|---:|---:|---:|---:|---:|
| 9 | 1 | 345.181 ms | 36.768 ms | 9.39x | 26.29 |
| 9 | 8 | 2,834.398 ms | 163.544 ms | 17.33x | 48.62 |
| 26 | 1 | 1,001.904 ms | 72.628 ms | 13.79x | 13.67 |
| 26 | 8 | 8,827.401 ms | 463.136 ms | 19.06x | 16.52 |
| 66 | 1 | 2,732.832 ms | 190.616 ms | 14.34x | 5.11 |
| 66 | 8 | 20,398.461 ms | 1,408.760 ms | 14.48x | 5.65 |

The optimized run retained all 197 Parameter storages. RSS stabilized at
381,100,032 bytes for 25 additional forwards. Full embedding parity passed
with maximum absolute/relative error 9.685754776000977e-8 /
2.95461479292632e-5; all 13 reference hidden states also remained inside the
unchanged MODEL-R3 tolerance.

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
- `TRANSFORMER_BENCH_MAX_M`: optional upper bound for M in the crossover sweep;
- `TRANSFORMER_BENCH_PROJECTION`: optional crossover projection filter such as
  `3072x768`. Omitting both filters preserves the complete historical sweep.

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

### Experimental numerical execution

The proposed bridge is now implemented inside the experimental graph module,
but remains disconnected from Tensor, FFI and PHP. Simulation records a plan
without buffers; numerical execution creates real initialized `Vec<f32>` slots,
validates an `ExecutionPlan`, and calls the existing kernels.

The physical pool tracks each slot as `Available`, `Borrowed(NodeId)`,
`Bound(ValueId)`, `PinnedExternal(ValueId)`, or `Delivered`. Acquiring a
destination moves its Vec into a `DestinationLease`; this permits immutable
borrows of other input slots and one exclusive output without unsafe aliasing.
Only a successful kernel call commits the lease. An error returns the Vec to
`Available` without publishing its partial contents. External inputs never
enter the pool, and external outputs remain pinned until ownership is delivered
in the `ExecutionResult`.

Bitwise parity against the direct allocating chain passes for all 24 established
shapes (`M=1..128` across `768->768`, `768->3072`, and `3072->768`). Tests also
execute branching, multiple external outputs, invalid plans, insufficient
capacity, incompatible dtype/layout/shape, and aborted leases.

For `[128,768]`, real pool metrics are:

| Metric | Allocating chain | Graph execution |
|---|---:|---:|
| logical outputs | 4 | 4 |
| physical allocations | 4 | 2 |
| slot reuses | 0 | 2 |
| releases before delivery | 3 | 3 |
| peak live slots | 2 | 2 |
| peak live output bytes | 786,432 | 786,432 |
| accumulated allocated output bytes | 1,572,864 | 786,432 |

Prompt destruction gives the baseline the same two-buffer live peak; the
measured benefit is fewer physical allocations and half the accumulated output
allocation, not a lower steady-state peak.

Run the numerical A/B benchmark with OpenBLAS fixed to one thread:

```bash
TRANSFORMER_GRAPH_BENCH_SAMPLES=25 \
TRANSFORMER_GRAPH_BENCH_TARGET_US=5000 \
    cargo bench --manifest-path runtime/Cargo.toml --bench graph
```

The hardening run (25 paired samples, OpenBLAS pthreads/Haswell, one thread)
produced:

| Variant | Median | p95 | Matmul | Add | Softmax | Transpose |
|---|---:|---:|---:|---:|---:|---:|
| current allocating chain | 3157.522 us | 3605.707 us | 1840.491 us | 46.208 us | 731.305 us | 332.175 us |
| graph executor | 3149.684 us | 3558.898 us | 1619.977 us | 35.884 us | 662.545 us | 328.767 us |

Samples use equal iteration counts and alternate A/B order; both variants
include identical per-kernel timing instrumentation. Earlier controlled runs
also showed a lower executor median, with meaningful host variance. The final
run measured `plan_build=1.269 us` and `plan_validate=0.064 us`, outside
steady-state execution. Diagnostic acquire/release measured 0.200/0.000 us
(the latter below timer resolution). Stage values
come from one diagnostic execution, so they identify instrumentation coverage
but do not sum to or statistically represent the paired median. The executor
is not in the production path and no production speedup is claimed.

The graph result is delivered as an owned Rust Vec without materialization.
`Float32Buffer` and `toFloat32()` remain PHP/FFI operations and cannot consume a
graph result while this executor is isolated. Their current baseline costs stay
in `profile_pipeline.php`. In the accompanying 15-sample run,
`native_without_export` was 5570.981 us, `export_final_buffer` was 89.957 us,
and `materialize_final`/`toFloat32()` was 5379.582 us. An executor A/B for those
modes is intentionally blocked until a separately approved FFI integration
exists; these baseline-only PHP figures must not be interpreted as graph
results.

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
| Graph execution | internal numerical executor validated bitwise; two real Vec slots, two reuses; paired median 3.94-3.99 ms on the reference run | local heuristics and refcount inference rejected | keep isolated; evaluate FFI integration only after explicit approval | physical reuse and lifecycle are proven, but PHP/FFI integration and production speedup are not claimed |

This matrix is the project record for previously evaluated optimizations. New
work should update measured values and decisions instead of reopening rejected
approaches without new evidence.

## Graph Executor — Hardening Report

The numerical executor remains an internal Rust experiment. It does not alter
Tensor, Storage, PHP, FFI, ABI, kernels, the matmul dispatcher, or OpenBLAS.
`GraphBuilder` constructs a topologically ordered DAG, lifetime analysis records
all consumers and the last use, the deterministic greedy planner assigns
physical slots, `ExecutionPlan` freezes node/value bindings, and execution calls
the existing kernels through exclusive destination leases.

### Audit and invariants

| Area | Hardened invariant |
|---|---|
| ownership / aliasing | A slot's Vec is moved into exactly one `DestinationLease`; inputs are immutable and occupy other live slots or external slices. No `unsafe` is used. |
| lifecycle | `Available -> Borrowed(NodeId) -> Bound(ValueId)`. A normal temporary then returns to `Available`; an external result follows `Bound -> PinnedExternal -> Delivered`. Every method checks its source state. |
| last use | closed lifetime intervals overlap at the consuming node. A value cannot share a slot with an output produced by its final consumer. Branches retain their source through the last branch. |
| value/slot binding | every produced value has one assignment; slot id, membership, dtype, layout, capacity and pairwise non-overlap are validated. |
| node/operation binding | node ids, operation, ordered inputs and output must exactly match the finalized graph. A producer must precede every consumer. |
| rollback | a kernel error or caught panic aborts the lease, restores its Vec and makes the slot available without publishing a Value. A test-only, thread-local failure seam does not modify kernels. |
| publication | commit occurs only after kernel success. External outputs are pinned after commit and can be delivered once. Partial buffers never become readable values. |
| external inputs | external slices receive no pool assignment, are borrowed immutably, validated before pool allocation, and are never released or mutated. |
| external outputs | pinned lifetimes extend to graph end; normal last-use release skips them and no later value may reuse their slot. |
| capacity / metadata | physical capacity may exceed logical `numel`, but it must never be smaller; matching dtype and contiguous row-major layout remain mandatory. Shape determines only the exact slice passed to a kernel. |
| malformed state | fallible graph analysis rejects invalid ids, producer links and non-topological dependencies. Storage and execution validation reject missing, out-of-range or altered assignments. |
| double actions | double acquire/release/delivery and delivery before publication return errors. A second commit/abort of the same lease is structurally impossible because the operation consumes the lease. |

The tests cover the requested lifetime families: diamond (`X -> A/B -> C`),
linear (`X -> A -> B -> C`), three consumers of X, two-source convergence,
and shared inputs crossing branches. They assert producer, ordered consumers,
consumer count, last use, lifetime boundaries, slot equality after legal reuse,
and slot inequality while intervals overlap. Numerical branching executes both
branches and compares the merge bit-for-bit with direct kernels.

External-output coverage includes final-only, intermediate, two-output and
all-output graphs, including an external intermediate with a later internal
consumer. State-machine tests cover valid transitions and attempts to acquire
an occupied/delivered slot, release twice, release a pinned result, or deliver
an unpublished result. Metadata tests cover exact, greater and insufficient
capacity plus incompatible dtype/layout. The established 24 Transformer shapes
retain bitwise parity and an additional case exercises zero, negative, very
large and very small values with varied residuals and stable softmax.

### Determinism, repetition, and complexity

Planning uses ordered Vec traversal and no hash iteration. Rebuilding and
planning the same graph 100 times produces identical graphs, lifetimes, slot
assignments, execution bindings, simulation events, allocation/reuse counts and
peak live slots.

Repeated numerical execution is deterministic, but the current pool belongs to
one `execute` call. Therefore every execution truthfully reports two physical
allocations and two intra-execution reuses for the four-node Transformer chain;
warm executions do **not** report zero new allocations. A persistent executor
would need to own a validated plan and an idle pool across calls, reset every
non-delivered slot after output transfer, and define concurrency/reentrancy and
failure recovery before it could claim zero-allocation steady state.

Lifetime analysis is linear in nodes plus edges. The greedy storage planner is
worst-case quadratic in produced values because each candidate scans existing
slots and their assigned lifetimes. Execution-plan validation also checks
pairwise lifetimes per slot and is worst-case quadratic. The graph benchmark
measures construction, lifetime analysis, storage planning, plan validation and
simulation separately for 10, 100, 1,000 and 10,000 nodes. General graph
coloring is intentionally out of scope unless these release measurements show
planning is material relative to numerical execution.

The 25-sample hardening run measured median/p95 microseconds:

| Nodes | Lifetime | Storage plan | Plan validation | Simulation |
|---:|---:|---:|---:|---:|
| 10 | 0.083 / 0.136 | 0.297 / 0.986 | 0.178 / 0.305 | 1.618 / 2.188 |
| 100 | 0.855 / 1.597 | 6.666 / 8.095 | 7.279 / 9.208 | 14.779 / 21.722 |
| 1,000 | 14.221 / 23.016 | 563.451 / 752.942 | 710.444 / 1344.213 | 179.206 / 348.260 |
| 10,000 | 263.345 / 937.442 | 124634.464 / 242702.406 | 88161.950 / 98993.858 | 1946.157 / 3447.463 |

Planning is negligible for the four-node numerical pipeline, but its quadratic
cost is material at 10,000 nodes. This does not justify graph coloring now;
before accepting graphs of that scale, the planner and pairwise validator need
an algorithmic optimization or an explicit supported-size limit.

### Reproducible benchmark

```bash
TRANSFORMER_GRAPH_BENCH_SAMPLES=25 \
TRANSFORMER_GRAPH_BENCH_TARGET_US=5000 \
OPENBLAS_NUM_THREADS=1 \
    cargo bench --manifest-path runtime/Cargo.toml --bench graph
```

The numerical section alternates paired A/B samples with a shared calibrated
iteration count. `plan_build` and `plan_validate` are outside the steady-state
pipeline measurement. Kernel, acquire and release fields come from one
diagnostic execution and must not be summed or presented as its statistical
median. Allocation/reuse columns are explicitly per execution. The current
baseline remains the allocating direct-kernel chain; no production speedup is
claimed.

Remaining risks are the absence of a persistent pool, lack of concurrent or
reentrant execution semantics, panic-hook output despite panic capture, no
allocator-level leak instrumentation, and no production Tensor/Storage
ownership bridge. These are integration blockers or future design work, not
results hidden by the experiment.

## Production Integration Gate

The next integration stage may be proposed only when all applicable boxes are
supported by tests and measurements:

- [x] graph, lifetime, storage and execution plans reject malformed input;
- [x] deterministic plan and slot selection;
- [x] no `unsafe`, premature recycle, double release or partial publication;
- [x] external inputs remain immutable and external outputs remain pinned;
- [x] controlled kernel error and panic rollback permit a later clean run;
- [x] branching and all external-output combinations execute correctly;
- [x] all 24 Transformer shapes retain bitwise parity;
- [x] reproducible release A/B and 10/100/1,000/10,000-node planner benchmark;
- [x] no public ABI/API, Tensor, Storage, FFI, PHP, kernel or dispatcher change;
- [ ] persistent-pool ownership, reset, concurrency and reentrancy contract;
- [ ] allocator/leak tooling over long repeated runs;
- [ ] production Tensor/Storage bridge design and independent approval;
- [ ] end-to-end production benchmark after that bridge exists.

Consequently the internal hardening gate is satisfied, while production
integration remains deliberately blocked on the unchecked items.

## Individual Operations — Performance Audit

This audit uses release builds, OpenBLAS 0.3.26 pthreads/Haswell fixed to one
thread, 25 samples, calibrated equal iteration counts and alternating A/B order
for candidates. Experimental code exists only in `benches/kernels.rs`; no
production kernel, dispatcher, Tensor, Storage, FFI, ABI or PHP API changed.

The preallocated native pipeline (`128x768`, `K=N=768`) measured 2715.156 us:

| Operation | Median | Pipeline fraction | Maximum gain if eliminated |
|---|---:|---:|---:|
| Matmul | 1632.021 us | 60.11% | 60.11% |
| Softmax, 128 rows | 688.425 us | 25.35% | 25.35% |
| Transpose | 353.077 us | 13.00% | 13.00% |
| Add | 41.633 us | 1.53% | 1.53% |

These fractions are an Amdahl bound for this pipeline and environment. They do
not transfer unchanged to PHP array creation, where input copy-in and final
zval materialization dominate.

### A. Matmul

The scalar loop reads B down columns and achieves only about 0.26–1.22 GFLOP/s.
The cache-friendly loop changes traversal to contiguous B rows and is strongly
autovectorizable; tiled improves larger M but remains far below SGEMM. No
backend copies or converts data: OpenBLAS consumes and produces contiguous
row-major buffers directly with `beta=0`.

Measured winners for every requested M:

| Projection | M=1 | M=2 | M=4,8,16,32,64,128 |
|---|---|---|---|
| 768→768 | cache-friendly | tiled | OpenBLAS/dispatcher |
| 768→3072 | cache-friendly | OpenBLAS/dispatcher | OpenBLAS/dispatcher |
| 3072→768 | cache-friendly | OpenBLAS/dispatcher | OpenBLAS/dispatcher |

At `128x768 * 768x768`, BLAS measured 1733.553 us and 87.101 GFLOP/s;
the dispatcher measured 1705.155 us in the independent sweep. At
`128x768 * 768x3072`, BLAS measured 7249.574 us and 83.312 GFLOP/s. At
`128x3072 * 3072x768`, BLAS measured 7169.705 us and 84.241 GFLOP/s.
Independent winner labels sometimes alternate between direct BLAS and the
dispatcher because of cache/noise; they execute the same SGEMM.

Shape selection alone costs 2–3 ns median. In paired execution the largest
measured dispatcher ratio was 1.0122 (`2x768 -> 3072`, 14.620 us), while the
other cases ranged from 0.9805 to 1.0058. At the representative M=128 case the
ratio was 1.0019. Repeated selection, validation and dynamic BLAS lookup are
therefore below the 5% rejection threshold. Caching backend decisions cannot
materially improve this pipeline and would add invalidation/configuration
state. Current dispatch policy is retained.

The output allocation is outside the pure kernel but required by the immutable
Tensor API. The Graph experiment proved that reducing four output allocations
to two changes the full numerical pipeline only from 3157.522 to 3149.684 us.
Matmul is compute-bound under BLAS; more Rust tiling, Rayon or manual SIMD is
not a competitive next step for calibrated shapes. Multi-threaded BLAS remains
an open deployment-level experiment, not a code change.

### B. Softmax and last-dimension batching

`softmax_f32` performs four logical passes: finite validation, maximum, fused
exponential/store/sum, and normalization. Subtracting the maximum preserves
stability. Exponential evaluation dominates; normalization cannot be fused
with the unknown final sum without recomputation or different storage.

| Case | Median | p95 | Logical GB/s |
|---|---:|---:|---:|
| rank-1, N=768 | 5.745 us | 6.714 us | 1.069 |
| 128 rank-1 rows | 690.888 us | 743.359 us | 1.138 |
| one last-dim call | 702.539 us | 773.671 us | 1.119 |

The batched Rust kernel deliberately calls the unchanged rank-1 implementation,
so its kernel time is equivalent rather than faster. Its benefit is structural:
the FFI comparison reduced 128 calls/allocations to one and measured 813.822 us
versus 725.012 us. Combining finite validation with the maximum pass is a
plausible benchmark-only hypothesis. SIMD needs a vector exponential with an
explicit accuracy/non-finite contract; row-level parallelism needs enough rows
to amortize scheduling. Neither is accepted without numeric and pipeline A/B.

### C. Transpose

The current loop reads source rows contiguously but writes the destination with
a stride of `rows`, causing write-allocate traffic and poor cache locality. It
logically reads and writes `4 * rows * columns` bytes each. A benchmark-only
tiled implementation was tested bit-for-bit with tiles 8, 16, 32 and 64.

| Shape | Best tile | Baseline median/p95 | Candidate median/p95 | Kernel speedup |
|---|---:|---:|---:|---:|
| 128x768 | 16 | 361.365 / 398.708 us | 91.287 / 93.425 us | 3.959x |
| 768x768 | 16 | 2126.171 / 2277.251 us | 680.164 / 874.179 us | 3.126x |
| 768x3072 | 16 | 15995.901 / 17170.679 us | 3685.778 / 4489.479 us | 4.340x |
| 3072x768 | 8 | 9910.079 / 10454.782 us | 4473.856 / 5551.850 us | 2.215x |

Tile 64 is not generally best and regressed p95 in the first run. For the full
`Matmul -> Add -> Softmax -> Transpose` pipeline, repeated paired results were:

| Tile | Baseline median/p95 | Candidate median/p95 | Pipeline speedup | Decision |
|---:|---:|---:|---:|---|
| 8 | 2950.185 / 5023.813 us | 2791.349 / 4839.523 us | 1.057x | investigate |
| 16 | 2955.562 / 3249.178 us | 2745.186 / 3234.246 us | 1.077x | investigate |
| 32 | 2880.672 / 3155.822 us | 2702.765 / 2979.863 us | 1.066x | investigate |
| 64 | 2950.475 / 3341.237 us | 2758.229 / 3757.514 us | 1.070x | reject tile 64 p95 |

This is a strong kernel optimization but only a 5–10% pipeline candidate. It
is intentionally not in production. A follow-up should choose tiles by shape,
repeat across hosts and measure the Tensor pipeline before proposing a patch.

#### Pode eliminar o transpose?

Elimination has a higher theoretical ceiling than tiling because it removes
the buffer and all traffic. For this native pipeline that ceiling is 13.00%
(`353.077 / 2715.156`); with final PHP-array materialization it is only about
4.7% (`353.077 / 7453.925`).

The current answer is **no without an internal layout/view architecture**.
`Tensor` owns storage whose length exactly matches a contiguous shape and its
constructor always creates contiguous strides. There is no shared storage,
offset or view lifetime. Add and softmax consume flat/contiguous data; matmul
and OpenBLAS assume row-major leading dimensions. Returning only reversed
shape/strides would make existing consumers interpret memory incorrectly.

A future graph-only view could defer the physical transpose if every downstream
consumer accepts the strided layout or if layout propagation folds the
transpose into a later matmul transpose flag. That requires alias ownership,
offsets, stride-aware kernel contracts and materialization barriers. It cannot
be introduced as a metadata tweak and is outside this audit.

### D. Add

Add is one contiguous read of each input and one contiguous write. LLVM can
autovectorize the simple loop. It measured 25.142 us and 46.919 GB/s for 98,304
elements, or only 1.53% of the profiled native pipeline. Perfect elimination
could save at most 1.53%. Matmul+add fusion and naive in-place/storage reuse
already failed their pipeline gates. Add is classified as optimization-saturated.

### E–G. Dispatcher, FFI/lifecycle and operation interaction

The raw preallocated FFI and Tensor API medians quantify wrapper/lifecycle cost:

| Operation | Raw FFI | Tensor API | Difference |
|---|---:|---:|---:|
| Matmul | 1632.021 us | 1667.237 us | 35.216 us |
| Add | 41.633 us | 58.939 us | 17.306 us |
| Softmax last-dim | 725.012 us | 730.325 us | 5.313 us |
| Transpose | 353.077 us | 356.342 us | 3.265 us |

Creating three input tensors, including three data copies, measured 365.062 us;
seven handle destructions measured 6.615 us; three metadata queries measured
5.412 us; three preallocated copy-outs measured 60.343 us. Construction follows
`Vec -> Storage -> Tensor(shape + strides) -> Box<TransformerTensor>`.
Storage wrapping and Box creation move ownership without copying the elements;
the output Vec initialization/data copy is the dominant allocation work. The
current profiler cannot truthfully attribute allocator internals separately to
Vec, shape/stride Vecs and Box, so structural counts are reported rather than
invented timings.

For 98,304 elements, `toFloat32()` measured 4639.199 us, while
`exportFloat32Buffer()` measured 88.941 us. The former is dominated by PHP zval
creation and keeps its public contract; the latter remains the preferred binary
consumer path. Graph acquire/release measured 0.200/about 0.000 us,
plan validation 0.064 us and plan build 1.269 us for four nodes. Allocation
reuse is real but kernels mask it, and its complete A/B is effectively tied.

Changing operation order is not semantics-preserving: residual addition must
precede softmax, and softmax does not commute with transpose unless the reduced
axis is also changed. Matmul+add fusion remains rejected; batching softmax is
accepted and already present. No further cross-operation change meets the gate.

### Consolidated decision matrix

| Operation | Baseline | Candidate | Speedup | Pipeline impact | p95 | Decision |
|---|---:|---:|---:|---:|---:|---|
| Matmul dispatcher | direct selected backend | repeated dispatch | 0.988–1.020x | below 1.3% | no consistent regression | keep current |
| Softmax batching | 128 FFI calls, 813.822 us | one call, 725.012 us | 1.123x | removes 127 calls/results | better median | accepted/current |
| Tiled transpose | 361.365 us | tile 16, 91.287 us | 3.959x kernel | 7.7% pipeline | 93.425 us kernel; pipeline comparable | investigate, benchmark-only |
| Add | 41.633 us | perfect elimination bound | at most 1.016x pipeline | 1.53% | n/a | saturated |
| Graph Executor | 3157.522 us | 3149.684 us | 1.002x | 0.25% | 3558.898 vs 3605.707 us | experimental |
| Matmul+add fusion | allocating pair | fused experiment | inconsistent | max projected ~2.7% | inconsistent | rejected |
| Naive storage reuse | four output allocations | two allocations | inconsistent | regressions to ~9.3% | often worse | rejected |
| Materialization | `toFloat32`, 4639.199 us | Float32Buffer, 88.941 us | 52.16x for export | consumer-dependent | 103.682 us buffer | both retained |

| Method | Bottleneck? | Current state | Next action |
|---|---|---|---|
| `matmul_f32` | yes if selected | reference correctness baseline | retain |
| cache-friendly matmul | yes at M=1 | measured winner | retain |
| tiled matmul | yes at 768→768 M=2/fallback | measured winner/fallback | retain |
| OpenBLAS SGEMM | dominant compute | winner at calibrated thresholds | retain; optionally profile thread counts |
| `matmul_dispatch_f32` | no overhead bottleneck | 2–3 ns selection | retain policy |
| `softmax_f32` | yes | stable four-pass rank-1 | benchmark combined validation/max and vector exp separately |
| `softmax_last_dim_f32` | yes | one FFI call, same row kernel | retain batching |
| `transpose_f32` | yes | cache-unfriendly physical copy | continue tiled A/B; do not ship yet |
| `add_f32` | no | contiguous/autovectorizable | saturated |
| lifecycle/handles | no in resident compute | destruction ~6.6 us total | retain contracts |
| `Float32Buffer` | solves binary export | experimental API retained | use for native/binary consumers |
| `toFloat32()` | yes for PHP arrays | required compatibility path | retain contract |
| Graph Executor | no measured speed gain | real reuse, tied pipeline | remain isolated |
| storage reuse | no | naive approach rejected | require new ownership model |
| matmul+add fusion | no | rejected | do not reopen without new hypothesis |

### Ranked bottlenecks and recommendation

1. Matmul/SGEMM: 60.11% of the native numerical pipeline.
2. PHP array materialization: 4639.199 us when an array is requested.
3. Softmax last-dim: 25.35% of native numerical execution.
4. Physical transpose: 13.00%; tiled kernel promising, pipeline gain 5–10%.
5. Input PHP-array copy-in: 365.062 us at Tensor API level and much larger when
   PHP creates/traverses the source arrays.

Accepted behavior remains OpenBLAS dispatch, M=1 cache-friendly matmul,
last-dimension batching, the single-guard array conversion and Float32Buffer.
Rejected work remains matmul+add fusion and naive storage reuse. Open hypotheses
are tiled transpose portability, combining softmax validation with maximum,
vector exponential accuracy/performance, row-parallel softmax, deployment BLAS
threading, and future graph-wide layout propagation.

The objective next step is a second, cross-host tiled-transpose experiment with
shape-dependent tile selection and a Tensor-level paired pipeline benchmark.
The present 5.7–7.7% pipeline gain is evidence to investigate, not enough to
change production under the 10% acceptance rule.

## Exclusive Tiled Transpose Experiment — Final Decision

The follow-up remained benchmark-only and added tiles 4 and 128, p99, odd/small
shape parity, an adaptive-selection model, and an internal Tensor lifecycle
model. The host was an Intel i3-9100F under a four-vCPU virtualized environment,
OpenBLAS remained single-threaded, and the repeat used `taskset -c 3`. System
load was 2.59 during the first extended run. Affinity did not remove periodic
latency modes, so reproducibility and p95 are part of the rejection rather than
being hidden by a favorable median.

### Kernel results for every tile and shape

The table reports the fixed-affinity repeat as candidate median/p95 in
microseconds followed by baseline/candidate speedup. Every candidate passed
bitwise validation before timing.

| Shape | Tile | Candidate median | Candidate p95 | Median speedup | p95 speedup |
|---|---:|---:|---:|---:|---:|
| 128x768 | 4 | 285.752 | 295.917 | 2.983x | 3.644x |
| 128x768 | 8 | 257.128 | 273.177 | 3.288x | 3.818x |
| 128x768 | 16 | 254.756 | 280.664 | 3.367x | 3.857x |
| 128x768 | 32 | 93.550 | 109.551 | 3.710x | 51.050x* |
| 128x768 | 64 | 266.992 | 296.768 | 3.341x | 3.775x |
| 128x768 | 128 | 559.429 | 870.640 | 1.580x | 1.595x |
| 768x768 | 4 | 2069.624 | 3349.251 | 2.596x | 2.028x |
| 768x768 | 8 | 800.997 | 10052.425 | 2.706x | 1.225x |
| 768x768 | 16 | 1930.405 | 3097.542 | 2.873x | 2.270x |
| 768x768 | 32 | 1039.902 | 11048.029 | 2.023x | 1.135x |
| 768x768 | 64 | 1727.305 | 11705.531 | 1.234x | 1.080x |
| 768x768 | 128 | 1742.605 | 13060.636 | 1.230x | 0.995x |
| 768x3072 | 4 | 17075.685 | 27907.115 | 2.719x | 1.813x |
| 768x3072 | 8 | 15943.937 | 31991.691 | 3.102x | 2.445x |
| 768x3072 | 16 | 14538.635 | 15801.738 | 3.293x | 3.826x |
| 768x3072 | 32 | 14278.836 | 16113.142 | 3.299x | 3.664x |
| 768x3072 | 64 | 17315.347 | 27328.208 | 2.645x | 2.149x |
| 768x3072 | 128 | 16861.546 | 27292.114 | 2.814x | 2.281x |
| 3072x768 | 4 | 15081.538 | 16391.240 | 1.839x | 1.867x |
| 3072x768 | 8 | 15012.938 | 16657.343 | 2.088x | 2.854x |
| 3072x768 | 16 | 17979.648 | 28770.676 | 1.626x | 1.054x |
| 3072x768 | 32 | 17096.738 | 26855.502 | 1.658x | 1.170x |
| 3072x768 | 64 | 17632.137 | 30304.792 | 1.696x | 1.081x |
| 3072x768 | 128 | 18446.144 | 28484.667 | 1.092x | 1.021x |

`*` The 51.050x p95 ratio is not a credible kernel effect: the paired baseline
entered the host's slow latency mode while the candidate did not. Similar
bimodality appears in several 768x768 p95 values. These measurements establish
that tiling improves locality, but cannot establish one universal tile from
tail latency. Tiles 16/32 generally lead for `rows <= columns`; tile 8 is often
better for `rows > columns`. This is a hypothesis, not a production rule.

### Complete pipeline gate

The complete pipeline was executed directly for every tile; these are not sums
of independent stages:

| Tile | Baseline median/p95 | Candidate median/p95 | Median speedup | p95 speedup |
|---:|---:|---:|---:|---:|
| 4 | 3071.708 / 12677.030 us | 2790.707 / 13008.430 us | 1.101x | 0.975x |
| 8 | 12434.531 / 14243.734 us | 2748.707 / 14045.435 us | 4.524x* | 1.014x |
| 16 | 3159.608 / 13344.633 us | 2945.407 / 13995.734 us | 1.073x | 0.953x |
| 32 | 12192.231 / 13761.435 us | 2755.907 / 13311.034 us | 4.424x* | 1.034x |
| 64 | 3527.409 / 13604.734 us | 3057.007 / 13344.534 us | 1.154x | 1.019x |
| 128 | 11968.330 / 13486.235 us | 2920.807 / 13506.836 us | 4.098x* | 0.998x |

The starred medians are invalid as acceptance evidence because the identical
baseline switched to the slow mode only in those pairs. The earlier independent
25-sample repeat was stable at 1.057x–1.077x. In the affinity repeat, tile 4
barely crossed 1.10x but regressed p95; tile 16 regressed p95 and stayed below
1.10x; tile 64 crossed the median gate but did not reproduce its earlier median
and had previously regressed p95. No tile satisfies all gates.

### Tensor/native model and selection cost

An internal benchmark reproduces the production lifecycle for each operation:
output Vec allocation, unchanged kernel, Shape/strides construction and Tensor
ownership transfer. It intentionally excludes the C call because exposing the
tiled candidate through FFI would alter the prohibited production surface.

Tensor-model median speedups varied from 0.972x to 1.160x in the credible
fast-baseline pairs, with p95 ratios from 0.978x to 1.327x. Several other pairs
again placed only the baseline in the slow host mode and reported false 4x
medians. Tile 64 produced 1.105x median but 0.978x p95; it therefore fails the
tail gate. The benefit does not survive Tensor lifecycle reproducibly.

The experimental shape rule (`rows > columns -> 8`, otherwise `16`) costs
1 ns median; selection overhead is negligible. It is not integrated because
the selected tiles are not statistically stable across runs/hosts and the
pipeline/Tensor gates failed.

### Correctness coverage

All six tiles are compared bit-for-bit with `transpose_f32` for square,
rectangular, Transformer, non-multiple and small dimensions including 1x1,
2x3, 3x5, 7x17 and 31x33. Inputs include zeros, negatives, ordinary values,
`1e20` and `1e-20`. No tolerance, unsafe, ABI or API change is involved.

### Secondary softmax experiment

The benchmark-only candidate combines finite validation and maximum search but
keeps the same maximum order, exponential/store/sum pass and normalization.
It preserves bitwise output and the NaN/+Inf/-Inf error contract.

| Layer | Baseline median/p95 | Candidate median/p95 | Median speedup | p95 speedup |
|---|---:|---:|---:|---:|
| kernel `128x768` | 1423.510 / 1985.536 us | 1466.744 / 2098.668 us | 0.971x | 0.946x |
| complete pipeline | 14093.924 / 29042.351 us | 13639.224 / 19669.034 us | 1.033x | 1.477x |

The kernel itself regressed and the noisy pipeline median improved only 3.3%.
The candidate is rejected and remains benchmark-only.

### Decision: REJECT

Tiled transpose is **not accepted for production** in this experiment:

1. stable complete-pipeline medians remained below 1.10x;
2. candidates that crossed 1.10x did not reproduce or regressed p95;
3. Tensor-model benefit was inconsistent and tile 64 regressed p95;
4. the best tile changes with orientation and host state;
5. selection is cheap, but its policy lacks stable evidence.

No production diff is proposed. `transpose_f32`, Tensor API, FFI, ABI, PHP,
OpenBLAS, matmul dispatcher and all prior decisions remain unchanged. The tiled
and combined-softmax implementations stay exclusively in the benchmark as a
record of the rejected experiments.

## Remaining Bottlenecks — Systematic Audit

Transpose tiling and combined finite-check/maximum softmax are closed as
production investigations. Their benchmark code and measurements are retained
only as a reproducible historical record; they are not candidates in the
ranking below and must not be rerun to reopen the decision without a genuinely
new hypothesis.

### Complete cost map

Three different paths must be kept separate:

| Path (`128x768`, K=N=768) | Median | What it includes |
|---|---:|---|
| raw preallocated FFI pipeline | 2715.156 us | four numeric calls and caller-owned C buffers |
| resident Tensor, no export | 2984.764 us | Tensor outputs, Vec allocation, metadata/ownership and four FFI calls |
| resident Tensor + Float32Buffer | 3064.458 us | resident path plus one 393,216-byte copy-out |
| resident Tensor + PHP array | 7453.925 us | resident path plus Float32 copy and 98,304 PHP zvals |
| first PHP/native pass | 76266.610 us | PHP-array traversal, three input copy-ins, native pipeline and lifecycle |

Within the raw numerical pipeline, measured medians were matmul 1632.021 us
(60.11%), softmax 688.425 us (25.35%), transpose 353.077 us (13.00%) and add
41.633 us (1.53%). These fractions describe numerical execution only. Array
materialization and first-pass input conversion are separate bottlenecks.

The measured Tensor wrapper deltas over raw FFI were 35.216 us for matmul,
17.306 us for add, 5.313 us for softmax and 3.265 us for transpose. Three input
Tensor creations including native copies cost 365.062 us after CData existed;
seven destructions cost 6.615 us, three metadata calls 5.412 us, and three
preallocated copy-outs 60.343 us. PHP-side creation/traversal raised the three
input stage to 67242.892 us.

The code-derived output lifecycle is:

```text
validate pointers/metadata
  -> Shape allocation and checked numel
  -> zero-initialized output Vec
  -> unchanged destination kernel
  -> Vec moved into Storage (no element copy)
  -> Shape moved/cloned and contiguous Strides created
  -> Tensor moved into Box<TransformerTensor>
  -> opaque handle returned through FFI
```

Storage/Tensor/Box transfers do not copy tensor elements. Each immutable
operation does allocate and initialize a new output Vec; shape and stride Vecs
also make small allocations. The current profiler cannot attribute global
allocator time independently to Vec, Shape, Strides and Box, so no invented
per-component timings are reported. The Tensor/raw deltas bound their combined
cost. Avoiding zero-initialization before an SGEMM with `beta=0` is conceptually
possible but would require a sound initialized-memory construction; output size
is only 393,216 bytes in the main case and all wrapper/allocation overhead is
far below the 10% pipeline gate.

Error branches are cold. On the success path, pointer, rank, dtype, shape,
length and C-integer conversions are O(rank) or constant. Matmul dispatch
selection costs 2–3 ns; `OnceLock` makes symbol discovery a one-time warmup
cost. OpenBLAS validates exact lengths then receives A, B and C directly as
row-major buffers with no memcpy or layout conversion.

### Matmul audit and structural margin

The established crossover remains unchanged:

| Projection | M=1 | M=2 | M>=4 |
|---|---|---|---|
| 768→768 | cache-friendly | tiled | OpenBLAS |
| 768→3072 | cache-friendly | OpenBLAS | OpenBLAS |
| 3072→768 | cache-friendly | OpenBLAS | OpenBLAS |

OpenBLAS reaches roughly 83–87 GFLOP/s for the M=128 projections on the
reference run. Scalar is retained only as correctness/unknown-shape fallback;
cache-friendly and tiled cover the measured small-M cases. Dispatcher overhead
was between noise and 1.22%, with no stable p95 regression. Recent fixed-affinity
runs on the virtualized host were bimodal and are explicitly invalid for new
thresholds; the calibrated policy is therefore not changed.

Outside SGEMM, shape handling, validation, dispatch, Box creation and output
ownership together account for only tens of microseconds. The remaining
structural question is repeated work inside the GEMM backend: thread scaling
and possible packing of a fixed weight matrix. OpenBLAS's public call used here
does not expose a persistent packed-B object, so packing cannot be isolated
from SGEMM with the current API. It requires a benchmark-only backend or a
library with an explicit prepack contract before any architectural proposal.

### Materialization and copies

For 98,304 elements:

| Operation | Median | Semantics |
|---|---:|---|
| Rust Tensor -> preallocated CData | 48.604 us | one memcpy-equivalent copy |
| `exportFloat32Buffer()` | 88.941 us | independent owned CData copy |
| `toFloat32()` | 4639.199 us | copy plus one PHP zval per element |
| Float32Buffer -> bytes | 33.366 us | binary string conversion after export |

There is no redundant intermediate copy between resident Tensor operations.
The final copy cannot be removed from `toFloat32()` because its contract returns
an independently owned PHP list. Float32Buffer is already the additive path for
binary/native consumers and avoids zval materialization, but deliberately owns
a copy so it cannot outlive borrowed Rust storage. A zero-copy borrowed export
would require a new shared-lifetime/ABI contract and saves only about 89 us in
the resident path; it cannot meet the 10% gate and is not recommended.

The largest avoidable conversion is not output export but repeatedly rebuilding
inputs from PHP arrays. Reusing already-created native Tensors changes no ABI
and is already possible at the API level. Its benefit must be evaluated in a
real repeated-layer/model workload rather than credited to a kernel.

### Softmax remaining margin

The rejected fused-check experiment is closed. The remaining cost is primarily
scalar `f32::exp`, followed by output normalization. A 768-element row takes
about 5.745 us in the stable run; 128 rows take 690.888 us. Last-dimension
batching already removes 127 FFI calls but intentionally retains the original
stable row kernel.

Softmax accounts for 25.35% of numerical time. By Amdahl's law, a softmax-only
candidate must accelerate the complete softmax by at least about 1.56x to make
the whole pipeline 1.10x faster. Memory-pass elimination alone is unlikely to
reach that threshold because exponential evaluation dominates. A vector-exp
implementation is the only remaining CPU hypothesis with a sufficient ceiling,
but it must specify accuracy, NaN/Inf behavior, normalization error, bitwise or
documented numerical parity, row sizes and p95. Row parallelism may help 128-row
batches but risks scheduling overhead and oversubscription with threaded BLAS.

### Future transpose-view specification

Transpose tiling stays rejected. Elimination through a view is a distinct,
future architectural hypothesis with a 13.00% removable-time fraction, equal
to a theoretical maximum pipeline speedup of about 1.149x in the raw path. It
cannot reach 10% when `toFloat32()` dominates.

A valid view design would require:

- Tensor: shared Storage ownership, byte/element offset, logical shape and
  arbitrary strides instead of exclusive exact-length contiguous Storage;
- Storage: alias-safe shared lifetime and explicit mutability rules;
- add: two stride-aware inputs and a defined contiguous/strided destination;
- softmax: an axis iterator supporting non-unit stride or a materialization
  barrier before last-axis reduction;
- matmul: recognize simple transpose views and map them to CBLAS transpose
  flags/leading dimensions; arbitrary strided matrices must materialize;
- transpose: create a metadata view by swapping shape/strides, not write data;
- FFI/export: materialize whenever the public contiguous contract is requested;
- Graph Executor: propagate layouts, prove view lifetime and place explicit
  materialization nodes before unsupported consumers.

For the current pipeline, transpose is final, so a binary consumer capable of
understanding strides could avoid 786,432 logical bytes of read/write traffic
and one output allocation. Existing PHP arrays and Float32Buffer require
contiguous transposed order and would simply move the materialization to export.
Ownership and kernel-contract complexity are high relative to the narrow 14.9%
raw upper bound, so views are a specification, not one of the next experiments.

### Graph Executor after allocation removal

The current executor creates its two-slot pool on every `execute`; each run
truthfully reports two physical allocations and two intra-run reuses. A
persistent executor would need to own the validated plan and pool across calls,
reset `Delivered` slots after ownership transfer, define whether output is
copied or leased, reject concurrent/reentrant use or synchronize it, recover
after error/panic, and isolate pools for different graphs/shapes.

For four nodes, plan build is 1.269 us, validation 0.064 us and measured
acquire/release about 0.200/near-0.000 us. Planning becomes material only at
large graphs: storage planning/validation medians were approximately
0.563/0.710 ms at 1,000 nodes and 124.634/88.162 ms at 10,000 nodes. Persistent
plans avoid repeating that cost, but the current four-node pipeline does not
need planner optimization.

The physical-reuse A/B changed 3157.522 us to 3149.684 us (1.002x), while
halving accumulated output allocations from four to two and leaving peak live
bytes at 786,432. Thus zero-allocation steady state has structural/lifecycle
value but no evidence of a 10% latency gain. Once allocations disappear, the
next bottlenecks remain the same kernels: SGEMM, exponential softmax and physical
layout conversion. Persistent pooling is not a performance priority unless a
different workload proves allocator pressure, memory fragmentation or p95 gain.

### High-impact hypothesis matrix

Only hypotheses with a plausible whole-pipeline ceiling of at least 10% remain:

| Hypothesis | Current affected cost | Theoretical ceiling | Complexity | Risk | Required experiment | Gate |
|---|---:|---:|---|---|---|---|
| OpenBLAS thread scaling | matmul 60.11% | up to 2.51x pipeline if matmul vanished; about 1.43x at 2x GEMM | low | oversubscription, small-M regression, p95 | paired full pipeline at 1/2/4 threads, every calibrated shape, warm/cold and fixed affinity | >=1.10x pipeline, stable p95, no small-M regression |
| resident repeated inference with fixed native inputs/weights | first-pass input stage 67.243 ms | far above 10% when inputs are currently rebuilt | low | benchmark may not model real application lifecycle | compare N repeated blocks with create-once versus recreate-each-time; report first-call and steady state separately | >=1.10x real repeated workload without semantic change |
| fixed-weight/prepacked GEMM backend | matmul 60.11% | up to matmul's 60.11% removable fraction | high | dependency/backend contract, memory, invalidation, portability | benchmark-only explicit B prepack; amortize pack over 1/2/8/32/128 calls and all projections | >=1.10x pipeline after including amortized pack, stable p95 |
| SIMD/vector-exp softmax | softmax 25.35% | maximum 1.34x if softmax vanished | medium/high | accuracy, non-finite behavior, CPU dispatch | scalar versus vector exp, then full softmax/pipeline/Tensor across row sizes and CPUs | softmax >=1.56x and pipeline >=1.10x with numerical contract |
| transpose view/layout propagation | transpose 13.00% | 1.149x raw pipeline; <1.10x array path | very high | aliasing, ownership, stride semantics, BLAS materialization | future graph-only prototype with explicit materialization counters | >=1.10x target consumer and no ownership/API regression |

### Ranked next experiments

At most three experiments are recommended:

1. **OpenBLAS 1/2/4-thread full-pipeline scaling.** It affects the dominant
   60.11%, requires no kernel rewrite, and is the quickest high-ceiling test.
2. **Create-once resident repeated inference, then fixed-weight prepack if
   residency is already representative.** This separates the measured 67 ms
   PHP input conversion from numerical steady state and determines whether a
   packed-weight backend is worth investigating.
3. **SIMD/vector-exp softmax prototype in benchmarks only.** Proceed only with
   an explicit numerical-error and CPU-feature contract; it must achieve at
   least 1.56x softmax speedup before full-pipeline consideration.

Do not spend further investigation budget on add, dispatcher caching, output
Vec/Box micro-allocations, zero-copy Float32Buffer borrowing, matmul+add fusion,
naive storage reuse, current Graph pooling for latency, transpose tiling or
finite-check/maximum fusion. Their measured or theoretical pipeline impact is
below the gate, inconsistent, or already rejected.

### Exact production state

Production remains: contiguous row-major Float32 Tensor with exclusive Storage;
immutable operations allocate owned outputs; `matmul_dispatch_f32` selects
cache-friendly M=1, tiled small-M fallback and optional OpenBLAS at calibrated
thresholds; stable scalar rank-1 and batched last-dimension softmax are intact;
physical scalar transpose and contiguous add are intact; Tensor ABI and PHP API
are unchanged; `toFloat32()` returns a PHP list; Float32Buffer remains an
independent-copy experimental export. Graph Executor, tiled transpose and all
rejected fusion/reuse/softmax candidates are absent from the production path.

## Controlled Final-Candidate Experiments

These experiments are benchmark-only. They do not change production kernels,
dispatch, Tensor/Storage, FFI, ABI, PHP API, or the Graph Executor.

### Reproduction protocol

The host was a four-vCPU virtual machine with OpenBLAS 0.3.26 (`pthreads`,
Haswell target). Every comparison used release code, fixed CPU affinity,
explicit OpenBLAS/OMP thread limits, a separate warmup, identical preallocated
inputs and outputs, alternating paired order, and 25 samples. The kernel runner
can select an experiment with `TRANSFORMER_BENCH_FILTER`:

```bash
taskset -c 0-3 env \
  TRANSFORMER_BENCH_FILTER=blas_threads \
  TRANSFORMER_BENCH_SAMPLES=25 \
  TRANSFORMER_BENCH_SAMPLE_MS=20 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  cargo bench --manifest-path runtime/Cargo.toml --bench kernels

taskset -c 3 env \
  TRANSFORMER_BENCH_FILTER=vector_exp \
  TRANSFORMER_BENCH_SAMPLES=25 \
  TRANSFORMER_BENCH_SAMPLE_MS=20 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  cargo bench --manifest-path runtime/Cargo.toml --bench kernels

taskset -c 3 env \
  TRANSFORMER_RESIDENT_SAMPLES=25 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  php -d xdebug.mode=off runtime/benches/resident_pipeline.php
```

The OpenBLAS runner changes its thread count explicitly at each measured phase;
the surrounding benchmark remains single-threaded. No Rayon or nested parallel
region is active. Reported values below are pipeline median/p95 in microseconds.
The 1T value is the baseline from the paired 1T/2T comparison. Because the host
changed performance modes during the run, the separate paired 1T/4T baseline
occasionally differed and was not substituted into this table.

### Experiment 1: OpenBLAS threads

| Projection | M | 1T median/p95 | 2T median/p95 | 4T median/p95 | Best median | Speedup vs 1T | Tail result |
|---|---:|---:|---:|---:|---|---:|---|
| 768→768 | 1 | 305.694/511.141 | 313.229/510.440 | 129.040/1082.351 | 4T | 2.369x | p95 regressed |
| 768→768 | 2 | 617.191/804.675 | 631.856/848.651 | 295.250/5522.952 | 4T | 2.090x | p95 severely regressed |
| 768→768 | 4 | 577.849/736.167 | 787.364/2299.376 | 1629.042/7650.494 | 1T | 1.000x | 2T/4T regressed |
| 768→768 | 8 | 755.333/942.486 | 820.781/1694.086 | 4402.861/10170.799 | 1T | 1.000x | 2T/4T regressed |
| 768→768 | 16 | 461.500/8696.995 | 363.300/15609.491 | 245.500/14103.807 | 4T | 1.880x | anomalous tails |
| 768→768 | 32 | 841.100/3559.551 | 1715.751/10874.904 | 6937.870/26448.134 | 1T | 1.000x | regressed |
| 768→768 | 64 | 2973.929/5029.183 | 5647.269/22390.964 | 6614.742/28878.363 | 1T | 1.000x | regressed |
| 768→768 | 128 | 2862.100/13149.505 | 2313.299/16149.197 | 14061.197/51039.691 | 2T | 1.237x | p95 regressed |
| 768→3072 | 1 | 1307.915/2405.329 | 1282.458/2234.901 | 1658.667/2178.201 | 2T | 1.020x | stable, below gate |
| 768→3072 | 2 | 1335.907/4014.441 | 1058.307/4247.054 | 840.400/6407.035 | 4T | 1.590x | p95 regressed |
| 768→3072 | 4 | 1200.108/1478.350 | 1097.836/1498.458 | 998.418/5440.702 | 4T | 1.202x | p95 regressed |
| 768→3072 | 8 | 1672.686/1911.136 | 1238.093/1861.586 | 1135.462/8381.395 | 4T | 1.473x | 2T stable; 4T tail regressed |
| 768→3072 | 16 | 2292.423/2727.734 | 1885.023/2827.434 | 1736.234/12697.994 | 4T | 1.320x | tails regressed |
| 768→3072 | 32 | 3721.017/6548.675 | 2836.758/3454.297 | 2645.231/17352.254 | 4T | 1.407x | 2T stable; 4T tail regressed |
| 768→3072 | 64 | 6684.804/9435.405 | 4910.603/10329.939 | 5264.636/20373.711 | 2T | 1.361x | p95 regressed |
| 768→3072 | 128 | 12413.199/14116.299 | 9324.800/11950.800 | 8454.800/37393.898 | 4T | 1.468x | 2T stable; 4T tail regressed |
| 3072→768 | 1 | 860.840/956.825 | 817.680/946.675 | 985.734/1451.267 | 2T | 1.053x | stable, below gate |
| 3072→768 | 2 | 1333.401/1784.501 | 1016.212/1445.989 | 853.649/11917.046 | 4T | 1.562x | 2T stable; 4T tail regressed |
| 3072→768 | 4 | 1178.332/1426.026 | 1004.663/1268.613 | 888.927/8595.625 | 4T | 1.326x | 2T stable; 4T tail regressed |
| 3072→768 | 8 | 1481.149/2228.058 | 999.149/1491.569 | 908.206/8370.421 | 4T | 1.631x | 2T stable; 4T tail regressed |
| 3072→768 | 16 | 1941.034/2779.018 | 1355.350/2571.901 | 799.350/8451.004 | 4T | 2.428x | 2T stable; 4T tail regressed |
| 3072→768 | 32 | 2812.464/5925.595 | 1946.248/7613.460 | 3171.817/14897.108 | 2T | 1.445x | p95 regressed |
| 3072→768 | 64 | 5260.435/13477.138 | 3670.201/10861.837 | 2165.150/5913.302 | 4T | 2.430x | p99 76351.878: anomalous |
| 3072→768 | 128 | 9023.553/12776.855 | 5785.652/9079.153 | 8481.996/49192.379 | 2T | 1.560x | 2T stable; 4T tail regressed |

The coefficient of variation frequently exceeded 0.5 and reached 3.39. Results
were bimodal and several 4T p95/p99 values were orders of magnitude worse than
their medians. There is no defensible universal `M >= X` crossover. Two threads
show credible regions for wide projections and 3072→768, but also regressions;
they require validation on bare metal and another CPU before a policy exists.

Decision: **2T — INVESTIGATE**. **4T — REJECT** on this host. No production
threshold or OpenBLAS configuration is changed.

### Experiment 2: resident inputs and weights

The benchmark uses the existing FFI only. “Input” includes the matmul input and
the residual add input; “weight” is B. Scenario order rotates on every sample to
avoid sequential thermal/frequency bias. Totals include create, repeated
execution, destroy, and one final export. Values are median/p95 microseconds.

| Model | Input recreate | Weight recreate | 1 execution | 10 executions | 100 executions | 1000 executions |
|---|---|---|---:|---:|---:|---:|
| A | yes | yes | 4181.294/12756.684 | 53336.522/84585.036 | 730223.500/955088.438 | 3410320.273/4282153.272 |
| B | no | yes | 3635.095/8358.190 | 53143.422/86833.737 | 690237.357/927036.502 | 3286016.420/3819624.327 |
| C | yes | no | 3808.595/13595.283 | 44638.519/71774.830 | 629328.341/847478.002 | 3200607.629/3725052.656 |
| D | no | no | 5193.294/14545.282 | 44374.019/91470.239 | 623209.548/845806.908 | 3011132.272/3490897.037 |

Initial PHP-to-CData fill was 51,502.336 us. First native call without export
was 4,390.995 us median and 11,153.786 us p95. At 100 executions, D versus A
improved median 1.172x and p95 1.129x. At 1000, it improved median 1.133x and
p95 1.227x, reducing amortized total from 3410.320 to 3011.132 us/execution.
One execution does not amortize residency, and the 10-execution p95 regressed.

Current B is contiguous row-major `[K,N]`; SGEMM is OpenBLAS CBLAS. OpenBLAS may
pack internally, but its CBLAS API exposes no portable persistent packed-B
object. External packing would be backend-specific, consume another packed
copy, and require invalidation on pointer, shape, dtype, or content changes.
Weight-only residency C gains 1.160x at 100 calls but only 1.066x at 1000, so
there is no evidence that an additional persistent packing layer crosses the
pipeline gate after construction and memory costs.

Decision: **resident input + weight — ACCEPT — CANDIDATE FOR INTEGRATION** for
repeated workloads of at least roughly 100 executions. **Prepacking — REJECT**
pending evidence from a backend with an explicit reusable packed-weight API.

### Experiment 3: AVX2/FMA vector-exp softmax

The benchmark-only candidate uses AVX2/FMA when detected and a scalar fallback
otherwise. It retains max subtraction and the existing non-finite checks as
separate passes. On this host AVX2 and FMA were available.

| Last dimension | M | Softmax median speedup | Softmax p95 speedup |
|---:|---:|---:|---:|
| 768 | 1 | 1.528x | 1.589x |
| 768 | 2 | 1.670x | 1.278x |
| 768 | 8 | 1.582x | 1.387x |
| 768 | 32 | 1.653x | 1.321x |
| 768 | 128 | 1.483x | 1.082x |

Across widths 128, 768, 2048, and 3072, isolated median speedup ranged from
1.411x to 1.798x. The required M=128/width=768 workload did not reach the 1.56x
kernel gate. Full-pipeline measurements were:

| Pipeline shape | Baseline median/p95 us | Candidate median/p95 us | Median/p95 speedup |
|---|---:|---:|---:|
| M128, K768, N768 | 2807.401/13095.003 | 2811.901/13072.203 | 0.998x/1.002x |
| M128, K768, N3072 | 32194.109/47387.813 | 33435.410/46816.613 | 0.963x/1.012x |
| M128, K3072, N768 | 17891.101/29733.502 | 18509.106/28762.302 | 0.967x/1.034x |

Finite normal and extreme inputs, zeros, negatives, requested ranks/dimensions,
NaN, +Inf, and -Inf were exercised. Error variants for non-finite inputs match
the baseline. Maximum observed pipeline absolute error was about 2e-9, relative
error 4.36e-7, maximum difference 6 ULP, and maximum row-sum error 4.292e-6.
The candidate is therefore not bitwise compatible, even though numerical error
is small. It neither reaches the softmax gate nor the pipeline gate.

Decision: **REJECT**. Do not integrate or reopen finite-check/max fusion.

### Final decision matrix and remaining bottlenecks

| Candidate | Pipeline impact | Evidence | Complexity | Risk | Decision |
|---|---|---|---|---|---|
| OpenBLAS 2T | shape-dependent, sometimes >10% | bimodal host; some stable wide shapes | low | oversubscription and tails | INVESTIGATE |
| OpenBLAS 4T | median gains offset by severe tails | p95/p99 regression and CV up to 3.39 | low | high tail latency | REJECT |
| Resident input + weight | 1.172x at N=100; 1.133x at N=1000 | paired end-to-end and p95 gains | low | lifecycle discipline | ACCEPT — CANDIDATE FOR INTEGRATION |
| Persistent B prepack | not demonstrated | weight residency alone falls below gate at N=1000 | high | backend coupling, memory, invalidation | REJECT |
| SIMD/vector-exp | 0.963x–0.998x pipeline median | isolated gain disappears end-to-end | medium/high | precision and CPU dispatch | REJECT |

Measured production-kernel ranking remains: matmul 60.11%, softmax 25.35%,
physical transpose 13.00%, and add 1.53%. For PHP/native end-to-end work, input
and weight lifecycle dominates cold/recreate paths and is the only candidate in
this round to pass the complete pipeline gate. Production state remains exactly
as documented above; ACCEPT here means candidate for a later, separately
audited integration only.

## Controlled Integration Proposal: Resident Inputs and Weights

This stage validates the previously accepted candidate without changing the
production path. The existing API already supports residence: callers create a
native Tensor once, retain its PHP `Tensor`/`NativeStorage`, and reuse the same
opaque handle in multiple immutable operations.

### Architecture and lifecycle

Current recreate path:

```text
PHP list -> temporary FFI CData -> copied Vec<f32> -> Storage -> Tensor
         -> Box<TransformerTensor> -> immutable kernel input
```

Proposed use of the unchanged path:

```text
setup:    PHP lists -> CData -> A/B/Residual native handles (one copy each)
bind:     validate Float32 + exact shape + contiguous row-major + capacity
execute:  borrow A/B/Residual as &[f32] -> allocate X/Y/Z/O separately
repeat:   borrow the same resident handles again
teardown: destroy O, then A/B/Residual exactly once
```

No `ResidentTensor` type is needed. Residency is a lifecycle policy over the
existing exclusively owned Tensor. `TransformerTensor` owns one Tensor, Tensor
owns one Storage, and Storage owns one Vec. Kernel calls receive immutable
slices; destinations are newly allocated, non-overlapping buffers. PHP
`NativeStorage` is the unique owner of its opaque handle and destruction is
idempotent on the PHP side. Use after destruction fails before FFI entry.

Resident A, B and residual must remain immutable and alive through every call.
Outputs X/Y/Z/O are never bound as resident inputs by the proposal. There is no
global resident registry, cache, mutable singleton or shared temporary state.
Two independent sets are therefore logically isolated and sequentially
reentrant. Parallel execution is not proposed; a future concurrency contract
would need to keep immutable inputs shareable while giving each invocation
exclusive output storage.

Safe Rust currently makes invalid dtype and non-contiguous layout
unrepresentable: `DType` contains only Float32 and Tensor always constructs
canonical row-major strides with storage length exactly equal to numel. The
benchmark binding nevertheless checks dtype, exact shape, canonical strides,
capacity and non-empty kernel requirements so the contract remains explicit if
Tensor gains new representations later.

### Internal allocation instrumentation

`benches/residency.rs` invokes the existing Tensor constructors, dispatcher and
kernels. Its counters surround every real Vec/Tensor construction in the
measured path. “Handle creations” models the one opaque FFI handle that the
existing Tensor API creates for each Tensor output; it does not instrument or
change the allocator. Counts are exact for the current four-operation pipeline:
three resident inputs plus four output buffers/Tensors/handles per inference.

```bash
taskset -c 3 env \
  TRANSFORMER_RESIDENCY_SAMPLES=25 \
  TRANSFORMER_RESIDENCY_MAX_N=1000 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  cargo bench --manifest-path runtime/Cargo.toml --bench residency
```

The run is release, single-core-affined, 25-sample, paired and alternating.
Model A recreates A/B/residual for every inference; model B creates them once
per measured N-inference session. Totals include setup and teardown to avoid
giving residence free lifecycle costs.

| N | A p50/p95/p99 us | B p50/p95/p99 us | p50 speedup | p95 speedup |
|---:|---:|---:|---:|---:|
| 1 | 12242.469 / 14707.262 / 15229.962 | 12551.968 / 14226.964 / 14846.762 | 0.975x | 1.034x |
| 10 | 94242.844 / 111111.736 / 130662.547 | 78618.859 / 107828.321 / 124318.411 | 1.199x | 1.030x |
| 100 | 977120.800 / 1085195.246 / 1142093.344 | 831330.454 / 866314.148 / 1025508.839 | 1.175x | 1.253x |
| 1000 | 3706714.603 / 9828388.949 / 9867516.503 | 3306928.131 / 8331562.061 / 8393939.983 | 1.121x | 1.180x |

The host remains bimodal at long N, but the paired median and both tail
indicators move in the same direction. Separate phase instrumentation measured
setup at 534.399 us median, teardown at 2.100 us, first execution at 3239.695 us
and second at 12086.882 us. The inverted first/second relationship and their
large p99 spread mark that phase run as **ANOMALOUS**; it is retained for
transparency and is not evidence for the decision.

| N | Model | allocations/Vecs | Tensor/handles | allocated bytes | copied input bytes |
|---:|---|---:|---:|---:|---:|
| 1 | A | 7 | 7 | 4,718,592 | 3,145,728 |
| 1 | B | 7 | 7 | 4,718,592 | 3,145,728 |
| 10 | A | 70 | 70 | 47,185,920 | 31,457,280 |
| 10 | B | 43 | 43 | 18,874,368 | 3,145,728 |
| 100 | A | 700 | 700 | 471,859,200 | 314,572,800 |
| 100 | B | 403 | 403 | 160,432,128 | 3,145,728 |
| 1000 | A | 7000 | 7000 | 4,718,592,000 | 3,145,728,000 |
| 1000 | B | 4003 | 4003 | 1,576,009,728 | 3,145,728 |

At N=1000 residence removes 42.8% of logical allocations/Tensor/handle
creations, 66.6% of allocated bytes and 99.9% of input-copy bytes. It does not
reuse X/Y/Z/O; the four output allocations per inference deliberately remain.

### Correctness and isolation gates

The benchmark asserts bitwise equality between first and second execution and
between two independently constructed resident sets. It also rejects wrong
shape, unsupported layout and insufficient capacity. Existing Rust and PHP
tests cover incompatible matmul/add shapes, non-finite softmax, null handles,
metadata, destruction idempotence and use after destruction. A destroyed PHP
Tensor cannot be bound because `NativeStorage::handle()` throws before FFI;
safe Rust cannot hold a reference after its Tensor is dropped.

The Graph Executor remains isolated. A future boundary can be:

```text
immutable resident Storage -> external ExecutionPlan inputs
                           -> per-execution temporary StoragePool -> output
```

Resident storage must never receive a pool slot or become a mutable destination.
No part of that bridge or the persistent Graph pool is implemented here.

### Production gate decision

Correction, bitwise parity, independent-set isolation, sequential reentrancy,
allocation reduction and regression gates pass. Performance passes at N=100
(1.175x median, improved p95) and N=1000 (1.121x median, improved p95). N=1
does not benefit, so residency must remain an explicit long-lived caller
lifecycle rather than hidden eager caching.

Decision: **ACCEPT — INTEGRAR**. This means the minimum later production change
is documentation and application lifecycle wiring that retains existing Tensor
objects; no native ABI/API or Tensor/Storage change is technically required.
No production integration or commit is performed in this stage.

## Production/Application Resident Lifecycle

The accepted lifecycle is now demonstrated by the real public PHP caller in
`examples/ffi/06-native-tensor-pipeline.php`. It creates input, weight and
residual once, captures those existing Tensor objects in the inference closure,
destroys X/Y/Z after every call, retains only the latest O, and destroys all
residents during final teardown. No Resident type or API was added.

The production/application confirmation uses `FfiBackend`, `NativeLibrary`,
PHP `Tensor` and `NativeStorage`; it does not call raw FFI functions. Run it as:

```bash
taskset -c 3 env \
  TRANSFORMER_RESIDENT_APPLICATION_SAMPLES=25 \
  TRANSFORMER_RESIDENT_APPLICATION_MAX_N=100 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  php -d xdebug.mode=off runtime/benches/resident_application.php
```

The default remains 25 samples and N through 1000. A minimum-N filter permits
an intentionally reduced long run without repeating smaller cases. The public
PHP recreate path is expensive: every input Tensor normalizes a PHP list,
creates CData, copies PHP→CData, calls FFI, then copies CData→Vec. Consequently
25 paired N=1000 samples take roughly one hour on this host.

The 25-sample application run produced:

| N | Recreate p50/p95/p99 us | Resident p50/p95/p99 us | p50 speedup | p95 speedup |
|---:|---:|---:|---:|---:|
| 1 | 144970.370 / 192602.058 / 197856.157 | 141056.770 / 206788.455 / 221121.458 | 1.028x | 0.931x |
| 10 | 1729006.040 / 1987558.255 / 2034648.216 | 235132.741 / 275413.846 / 289815.304 | 7.353x | 7.216x |
| 100 | 7300051.149 / 14744545.636 / 18805991.015 | 391809.106 / 949333.289 / 1157845.337 | 18.632x | 15.532x |

N=1000 was confirmed with three samples because a 25-sample recreate run was
operationally disproportionate. Its median was 76,265,935.891 us recreate and
3,266,057.486 us resident, a 23.351x speedup. The observed maxima were
139,726,861.407 and 3,381,642.187 us. With only three samples these values are
not statistically robust p95/p99 and are not presented as such. The earlier
25-sample FFI lifecycle benchmark remains the tail-latency gate for N=1000.

Resident N=100 phase medians were setup 68,709.196 us, steady execution
322,622.403 us and teardown 335.600 us. At N=1000 they were 71,206.289 us,
3,195,226.859 us and 582.100 us. Setup is never hidden from total comparisons.

Structural counts follow directly from the public implementation and are
secondary evidence, not allocator telemetry:

| N | Model | input Tensors | PHP→CData copies | CData→Vec copies | output Tensors | total Tensors/handles |
|---:|---|---:|---:|---:|---:|---:|
| 100 | recreate | 300 | 300 | 300 | 400 | 700 |
| 100 | resident | 3 | 3 | 3 | 400 | 403 |
| 1000 | recreate | 3000 | 3000 | 3000 | 4000 | 7000 |
| 1000 | resident | 3 | 3 | 3 | 4000 | 4003 |

Actual PHP, CData, Rust and OpenBLAS allocator calls are **not instrumented**.
No allocator precision is inferred from these structural counts. Output
allocations remain unchanged by design.

Application decision: **ACCEPT — INTEGRAR**. N=100 and N=1000 exceed the 1.10x
median gate, tail measurements do not regress, bitwise tests pass, and N=1 has
only a 2.8% median difference. The integration is additive usage/documentation;
Tensor, Storage, kernels, FFI and ABI remain unchanged.

## NN-1/NN-2 Linear Benchmark

`Linear` is the first real neural module. Weight and optional bias are resident
`Parameter` properties; input and output belong to each `forward()` call. The
native `linear_last_dim` operation flattens preceding dimensions conceptually,
uses the existing matmul dispatcher, and adds one unexpanded bias row-wise.

Reproduce with one OpenBLAS thread and fixed affinity:

```bash
taskset -c 3 env \
  TRANSFORMER_LINEAR_SAMPLES=25 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  php -d xdebug.mode=off runtime/benches/linear.php
```

Parameter creation, first forward, 25-sample steady-state forward, export, and
one independent PHP reference execution are timed separately. Weight creation
is never included in steady-state forward.

| Rank/M/K/N | Parameter setup | First forward | Steady p50/p95/p99 | Export | PHP reference | Max abs error |
|---|---:|---:|---:|---:|---:|---:|
| 2 / 1 / 768 / 768 | 238118 us | 233 us | 89 / 140 / 157 us | 36 us | 125480 us | 3.0e-8 |
| 3 / 8 / 768 / 768 | 239689 us | 605 us | 231 / 11793 / 12120 us | 222 us | 1084301 us | 1.4e-8 |
| 2 / 128 / 768 / 768 | 238188 us | 15903 us | 2547 / 14413 / 29280 us | 15433 us | 17359297 us | 1.4e-8 |
| 2 / 128 / 768 / 3072 | 998801 us | 47593 us | 33678 / 96242 / 109347 us | 109728 us | 36745480 us | 2.0e-8 |
| 2 / 128 / 3072 / 768 | 500542 us | 7213 us | 14508 / 19753 / 30981 us | 5709 us | 73766487 us | 5.3e-8 |

The host again showed bimodal tails, especially for M=8 and M=128, so p95/p99
are retained rather than using the best sample. Small deterministic integration
tests are bitwise equal to their independent PHP reference. Transformer-sized
cases differ by at most 5.3e-8 because PHP accumulates doubles while the native
dispatcher/OpenBLAS accumulates Float32; no tolerance is hidden in the report.

### Expanded Linear matrix and resident ownership

The required M sweep was run for every projection, with and without bias. The
backend column is `dispatcher`: M=1 reaches the cache-friendly policy and the
calibrated larger shapes reach OpenBLAS without copying or changing layout.

| M/K/N | No-bias p50/p95/p99 us | Bias p50/p95/p99 us | Paired bias median delta |
|---|---:|---:|---:|
| 1/768/768 | 125.802/300.506/11128.520 | 133.903/184.804/10639.910 | +8.101 us |
| 8/768/768 | 307.306/10577.508/11644.429 | 283.205/399.508/10878.415 | -24.101 us |
| 32/768/768 | 592.211/9952.796/10319.904 | 586.912/9757.693/9824.193 | -5.299 us |
| 128/768/768 | 1897.137/12333.739/12653.045 | 1816.935/12171.036/12828.849 | -80.202 us |
| 1/768/3072 | 891.691/12449.577/12609.175 | 896.891/3652.064/3935.862 | +5.200 us |
| 8/768/3072 | 1467.486/11732.185/11760.284 | 1488.685/11091.791/11853.383 | +21.199 us |
| 32/768/3072 | 4107.359/12950.768/13302.364 | 2660.173/8436.417/11922.979 | -1447.186 us |
| 128/768/3072 | 17185.625/27557.619/29464.400 | 17720.219/26921.320/26939.325 | +534.594 us |
| 1/3072/768 | 856.991/12062.672/12210.771 | 816.991/12172.572/12390.969 | -40.000 us |
| 8/3072/768 | 1616.982/13492.353/14004.647 | 1499.783/13849.049/29119.793 | -117.199 us |
| 32/3072/768 | 2793.270/13533.053/13763.250 | 2889.069/13237.955/13483.453 | +95.799 us |
| 128/3072/768 | 17405.304/27504.790/28590.578 | 17265.906/29239.870/29479.079 | -139.398 us |

The paired run alternates order and shares input/weight, but this virtualized
host remains strongly bimodal. Negative “bias costs”, especially
M32/768→3072, are physically impossible as kernel costs and mark unresolved
host modes rather than an optimization. The direct bias pass is O(MN), uses no
expanded allocation, and is small enough to fall below run-mode variance in
most cases. No fusion or dispatch change is accepted from these timings.

The full PHP-reference sweep covered M=1/8/32/128 for 768→768, 768→3072 and
3072→768 with both bias modes. Maximum absolute error was 6.05e-7 (M=1,
3072→768); other reported cases were at or below 5.3e-8. The wider error still
comes from PHP-double versus Float32 accumulation and remains small, but the
contract is not declared bitwise for OpenBLAS workloads.

Representative no-bias median throughput was 79.6 GFLOP/s for
128×768→768, 35.2 GFLOP/s for 128×768→3072, and 34.7 GFLOP/s for
128×3072→768. The benchmark prints GFLOP/s from `2*M*K*N` and steady p50;
bias additions are intentionally excluded from the GEMM FLOP convention.

Linear-owned parameter residence was measured independently with fixed input,
paired alternating order and five samples:

| N | Recreate p50/p95/p99 us | Resident p50/p95/p99 us | Median speedup | Weight/bias creations |
|---:|---:|---:|---:|---:|
| 1 | 108611.608/148555.575/148555.575 | 119228.766/147800.608/147800.608 | 0.911x | 1/1 vs 1/1 |
| 10 | 969467.344/1187615.486/1187615.486 | 99063.302/119704.348/119704.348 | 9.786x | 10/10 vs 1/1 |
| 100 | 12983367.288/13412487.377/13412487.377 | 169032.092/223673.065/223673.065 | 76.810x | 100/100 vs 1/1 |

N=1000 was not repeated: public PHP construction makes that recreate run
disproportionately expensive, and the earlier 25-sample Tensor/FFI residence
gate already established N=1000 behavior. This Linear run demonstrates the new
module ownership specifically; it is not used to reopen the accepted residence
decision.
