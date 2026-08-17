use std::env;
use std::hint::black_box;
use std::time::{Duration, Instant};

#[allow(dead_code, unused_imports)]
#[path = "../src/kernels/add.rs"]
mod add;
#[path = "support/blas.rs"]
mod bench_blas;
#[allow(dead_code, unused_imports)]
#[path = "../src/kernels/blas.rs"]
mod blas;
#[allow(dead_code, unused_imports)]
#[path = "../src/kernels/matmul.rs"]
mod matmul;
#[allow(dead_code, unused_imports)]
#[path = "../src/kernels/matmul_dispatch.rs"]
mod matmul_dispatch;
#[allow(dead_code, unused_imports)]
#[path = "../src/kernels/softmax.rs"]
mod softmax;
#[allow(dead_code, unused_imports)]
#[path = "../src/tensor/mod.rs"]
mod tensor;
#[allow(dead_code, unused_imports)]
#[path = "../src/kernels/transpose.rs"]
mod transpose;

mod kernels {
    pub(crate) use crate::matmul;
}

const BASELINE: &str = "current_scalar";
const DEFAULT_SAMPLE_TARGET_MS: u64 = 20;
const DEFAULT_SAMPLES: usize = 15;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Profile {
    Quick,
    Transformer,
    Crossover,
}

impl Profile {
    fn from_environment() -> Self {
        match env::var("TRANSFORMER_BENCH_PROFILE").as_deref() {
            Ok("quick") => Self::Quick,
            Ok("crossover") => Self::Crossover,
            Ok("transformer") | Err(_) => Self::Transformer,
            Ok(value) => {
                panic!(
                    "invalid TRANSFORMER_BENCH_PROFILE={value:?}; expected quick, transformer, or crossover"
                )
            }
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Quick => "quick",
            Self::Transformer => "transformer",
            Self::Crossover => "crossover",
        }
    }
}

#[derive(Clone, Copy)]
enum Throughput {
    Gigabytes(f64),
}

struct Measurement {
    median: Duration,
    p95: Duration,
    p99: Duration,
    coefficient_of_variation: f64,
    iterations: usize,
}

fn main() {
    let profile = Profile::from_environment();
    let filter = env::var("TRANSFORMER_BENCH_FILTER").unwrap_or_default();
    let samples = environment_usize("TRANSFORMER_BENCH_SAMPLES", DEFAULT_SAMPLES).max(3);
    let target = Duration::from_millis(environment_u64(
        "TRANSFORMER_BENCH_SAMPLE_MS",
        DEFAULT_SAMPLE_TARGET_MS,
    ));

    println!("Transformer Runtime kernel benchmarks");
    println!(
        "baseline={BASELINE} profile={} samples={samples} sample_target_ms={} filter={filter:?}",
        profile.name(),
        target.as_millis()
    );
    println!(
        "kernel,variant,case,median_us,p95_us,iterations,metric,value,speedup_scalar,speedup_tiled,winner"
    );

    if matches_filter(&filter, "matmul")
        || matches_filter(&filter, "fusion")
        || matches_filter(&filter, "lifecycle")
        || matches_filter(&filter, "blas_threads")
    {
        let blas_threads = environment_usize("TRANSFORMER_BLAS_THREADS", 1);
        let blas_info = bench_blas::configure(blas_threads);
        println!(
            "blas_backend=openblas parallel={} threads={} core={} config={}",
            blas_info.parallel, blas_info.threads, blas_info.core, blas_info.config
        );
        println!("blas_layout=row_major transpose_a=false transpose_b=false explicit_copy=false");
        if matches_filter(&filter, "matmul") {
            benchmark_matmul(profile, samples, target);
            benchmark_dispatch_overhead(samples, target);
        }
        if matches_filter(&filter, "fusion") {
            benchmark_matmul_add_fusion(samples, target);
        }
        if matches_filter(&filter, "lifecycle") {
            benchmark_storage_reuse(samples, target);
        }
        if matches_filter(&filter, "blas_threads") {
            benchmark_blas_thread_scaling(samples, target);
        }
    }
    if matches_filter(&filter, "softmax") {
        benchmark_softmax(profile, samples, target);
    }
    if matches_filter(&filter, "vector_exp") {
        benchmark_vector_exp_softmax(samples, target);
    }
    if matches_filter(&filter, "add") {
        benchmark_add(profile, samples, target);
    }
    if matches_filter(&filter, "transpose") {
        benchmark_transpose(profile, samples, target);
    }
}

struct OwnedBenchTensor {
    values: Vec<f32>,
    rows: usize,
    columns: usize,
}

fn benchmark_storage_reuse(samples: usize, target: Duration) {
    println!(
        "lifecycle,shape,variant,median_us,p95_us,kernels_us,lifecycle_us,speedup,vec_allocations,allocated_bytes,result_handles_created,peak_result_handles,peak_output_bytes"
    );
    for (k, n) in [(768, 768), (768, 3072), (3072, 768)] {
        for m in [1, 2, 4, 8, 16, 32, 64, 128] {
            let a = deterministic_values(m * k, 0.001);
            let b = deterministic_values(k * n, 0.002);
            let residual = deterministic_values(m * n, 0.003);
            validate_storage_reuse(&a, &b, &residual, m, k, n);

            let mut first = vec![0.0; m * n];
            let mut second = vec![0.0; m * n];
            let kernel_only = measure(samples, target, || {
                matmul_dispatch::matmul_dispatch_f32(&a, &b, &mut first, m, k, n).unwrap();
                add::add_f32(&first, &residual, &mut second).unwrap();
                softmax::softmax_last_dim_f32(&second, &mut first, n).unwrap();
                transpose::transpose_f32(&first, &mut second, m, n).unwrap();
                black_box(second[0]);
            });
            let (baseline, reused) = measure_pair(
                samples,
                target,
                || {
                    black_box(run_allocating_pipeline(&a, &b, &residual, m, k, n));
                },
                || {
                    black_box(run_reusing_pipeline(&a, &b, &residual, m, k, n));
                },
            );
            let bytes = m * n * size_of::<f32>();
            report_lifecycle(
                m,
                k,
                n,
                "baseline",
                &baseline,
                &kernel_only,
                1.0,
                4,
                bytes * 4,
                4,
                2,
                bytes * 2,
            );
            report_lifecycle(
                m,
                k,
                n,
                "storage_reuse",
                &reused,
                &kernel_only,
                baseline.median.as_secs_f64() / reused.median.as_secs_f64(),
                2,
                bytes * 2,
                1,
                1,
                bytes * 2,
            );
        }
    }
}

fn run_allocating_pipeline(
    a: &[f32],
    b: &[f32],
    residual: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Box<OwnedBenchTensor> {
    let mut matmul = Box::new(OwnedBenchTensor {
        values: vec![0.0; m * n],
        rows: m,
        columns: n,
    });
    matmul_dispatch::matmul_dispatch_f32(a, b, &mut matmul.values, m, k, n).unwrap();

    let mut added = Box::new(OwnedBenchTensor {
        values: vec![0.0; m * n],
        rows: m,
        columns: n,
    });
    add::add_f32(&matmul.values, residual, &mut added.values).unwrap();
    drop(matmul);

    let mut normalized = Box::new(OwnedBenchTensor {
        values: vec![0.0; m * n],
        rows: m,
        columns: n,
    });
    softmax::softmax_last_dim_f32(&added.values, &mut normalized.values, n).unwrap();
    drop(added);

    let mut transposed = Box::new(OwnedBenchTensor {
        values: vec![0.0; m * n],
        rows: n,
        columns: m,
    });
    transpose::transpose_f32(&normalized.values, &mut transposed.values, m, n).unwrap();
    transposed
}

fn run_reusing_pipeline(
    a: &[f32],
    b: &[f32],
    residual: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Box<OwnedBenchTensor> {
    let mut current = Box::new(OwnedBenchTensor {
        values: vec![0.0; m * n],
        rows: m,
        columns: n,
    });
    let mut scratch = vec![0.0; m * n];
    matmul_dispatch::matmul_dispatch_f32(a, b, &mut current.values, m, k, n).unwrap();

    add::add_f32(&current.values, residual, &mut scratch).unwrap();
    std::mem::swap(&mut current.values, &mut scratch);
    softmax::softmax_last_dim_f32(&current.values, &mut scratch, n).unwrap();
    std::mem::swap(&mut current.values, &mut scratch);
    transpose::transpose_f32(&current.values, &mut scratch, m, n).unwrap();
    std::mem::swap(&mut current.values, &mut scratch);
    current.rows = n;
    current.columns = m;
    current
}

fn validate_storage_reuse(a: &[f32], b: &[f32], residual: &[f32], m: usize, k: usize, n: usize) {
    let baseline = run_allocating_pipeline(a, b, residual, m, k, n);
    let reused = run_reusing_pipeline(a, b, residual, m, k, n);
    assert_eq!(
        (reused.rows, reused.columns),
        (baseline.rows, baseline.columns)
    );
    assert_eq!(
        reused.values, baseline.values,
        "storage reuse changed results"
    );
}

#[allow(clippy::too_many_arguments)]
fn report_lifecycle(
    m: usize,
    k: usize,
    n: usize,
    variant: &str,
    measurement: &Measurement,
    kernel_only: &Measurement,
    speedup: f64,
    allocations: usize,
    allocated_bytes: usize,
    handles: usize,
    peak_handles: usize,
    peak_bytes: usize,
) {
    let lifecycle = measurement.median.saturating_sub(kernel_only.median);
    println!(
        "storage_reuse,{m}x{k}_x_{k}x{n},{variant},{:.3},{:.3},{:.3},{:.3},{speedup:.3},{allocations},{allocated_bytes},{handles},{peak_handles},{peak_bytes}",
        measurement.median.as_secs_f64() * 1e6,
        measurement.p95.as_secs_f64() * 1e6,
        kernel_only.median.as_secs_f64() * 1e6,
        lifecycle.as_secs_f64() * 1e6,
    );
}

fn benchmark_matmul_add_fusion(samples: usize, target: Duration) {
    println!(
        "fusion,shape,median_us,p95_us,speedup,handles_before,handles_after,buffers_before,buffers_after,ffi_calls_before,ffi_calls_after"
    );
    for (k, n) in [(768, 768), (768, 3072), (3072, 768)] {
        for m in [1, 2, 4, 8, 16, 32, 64, 128] {
            let a = deterministic_values(m * k, 0.001);
            let b = deterministic_values(k * n, 0.002);
            let residual = deterministic_values(m * n, 0.003);
            validate_matmul_add_candidate(&a, &b, &residual, m, k, n);
            let (baseline, fused) = measure_pair(
                samples,
                target,
                || {
                    let mut intermediate = vec![0.0; m * n];
                    let mut output = vec![0.0; m * n];
                    matmul_dispatch::matmul_dispatch_f32(
                        black_box(&a),
                        black_box(&b),
                        black_box(&mut intermediate),
                        m,
                        k,
                        n,
                    )
                    .unwrap();
                    add::add_f32(
                        black_box(&intermediate),
                        black_box(&residual),
                        black_box(&mut output),
                    )
                    .unwrap();
                    black_box(output);
                },
                || {
                    let mut output = vec![0.0; m * n];
                    matmul_add_candidate_f32(
                        black_box(&a),
                        black_box(&b),
                        black_box(&residual),
                        black_box(&mut output),
                        m,
                        k,
                        n,
                    )
                    .unwrap();
                    black_box(output);
                },
            );
            println!(
                "baseline,{m}x{k}_x_{k}x{n},{:.3},{:.3},1.000,2,2,2,2,2,2",
                baseline.median.as_secs_f64() * 1e6,
                baseline.p95.as_secs_f64() * 1e6,
            );
            println!(
                "matmul_add,{m}x{k}_x_{k}x{n},{:.3},{:.3},{:.3},2,1,2,1,2,1",
                fused.median.as_secs_f64() * 1e6,
                fused.p95.as_secs_f64() * 1e6,
                baseline.median.as_secs_f64() / fused.median.as_secs_f64(),
            );
        }
    }
}

/// Benchmark-only fusion candidate. This intentionally remains outside the
/// production kernel graph unless the end-to-end acceptance threshold is met.
fn matmul_add_candidate_f32(
    a: &[f32],
    b: &[f32],
    residual: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), matmul::MatmulError> {
    let expected = m
        .checked_mul(n)
        .ok_or(matmul::MatmulError::DimensionOverflow)?;
    if residual.len() != expected || output.len() != expected {
        return Err(matmul::MatmulError::LengthMismatch);
    }

    matmul_dispatch::matmul_dispatch_f32(a, b, output, m, k, n)?;
    for (value, residual_value) in output.iter_mut().zip(residual) {
        *value += residual_value;
    }
    Ok(())
}

fn validate_matmul_add_candidate(
    a: &[f32],
    b: &[f32],
    residual: &[f32],
    m: usize,
    k: usize,
    n: usize,
) {
    let mut intermediate = vec![0.0; m * n];
    let mut expected = vec![0.0; m * n];
    let mut actual = vec![0.0; m * n];
    matmul_dispatch::matmul_dispatch_f32(a, b, &mut intermediate, m, k, n).unwrap();
    add::add_f32(&intermediate, residual, &mut expected).unwrap();
    matmul_add_candidate_f32(a, b, residual, &mut actual, m, k, n).unwrap();
    assert_eq!(actual, expected, "matmul + add candidate changed results");
}

fn benchmark_matmul(profile: Profile, samples: usize, target: Duration) {
    let cases = match profile {
        Profile::Quick => vec![(16, 16, 16), (32, 64, 32), (1, 768, 768)],
        Profile::Transformer => vec![
            (1, 768, 768),
            (128, 768, 768),
            (128, 768, 3072),
            (128, 3072, 768),
        ],
        Profile::Crossover => crossover_matmul_cases(),
    };

    for (m, k, n) in cases {
        let a = deterministic_values(m * k, 0.001);
        let b = deterministic_values(k * n, 0.002);
        let case = format!("{m}x{k}_x_{k}x{n}");
        let operations = 2.0 * m as f64 * k as f64 * n as f64;
        let variants: [(&str, Matmul); 5] = [
            ("current_scalar", matmul::matmul_f32),
            ("cache_friendly", matmul::matmul_cache_friendly_f32),
            ("tiled", matmul::matmul_tiled_f32),
            ("blas_sgemm", bench_blas::matmul_blas_f32),
            ("dispatcher", matmul_dispatch::matmul_dispatch_f32),
        ];
        validate_blas_result(&a, &b, m, k, n);
        let mut measurements = Vec::with_capacity(variants.len());

        for (variant, matmul) in variants {
            let mut output = vec![0.0; m * n];
            let measurement = measure(samples, target, || {
                matmul(
                    black_box(&a),
                    black_box(&b),
                    black_box(&mut output),
                    m,
                    k,
                    n,
                )
                .unwrap();
                black_box(output[0]);
            });
            measurements.push((variant, measurement));
        }

        let scalar_latency = measurements[0].1.median;
        let tiled_latency = measurements[2].1.median;
        let winner = measurements
            .iter()
            .min_by_key(|(_, measurement)| measurement.median)
            .expect("matmul variants are not empty")
            .0;
        for (variant, measurement) in measurements {
            report_matmul(
                variant,
                &case,
                measurement,
                operations,
                scalar_latency,
                tiled_latency,
                winner,
            );
        }
    }
}

fn benchmark_blas_thread_scaling(samples: usize, target: Duration) {
    println!("blas_threads,shape,candidate_threads,layer,baseline_median_us,baseline_p95_us,baseline_p99_us,candidate_median_us,candidate_p95_us,candidate_p99_us,median_speedup,p95_speedup,baseline_cv,candidate_cv");
    for (k, n) in [(768, 768), (768, 3072), (3072, 768)] {
        for m in [1, 2, 4, 8, 16, 32, 64, 128] {
            let a = deterministic_values(m * k, 0.001);
            let b = deterministic_values(k * n, 0.002);
            let residual = deterministic_values(m * n, 0.003);
            for candidate_threads in [2, 4] {
                let mut baseline_first = vec![0.0; m * n];
                let mut baseline_second = vec![0.0; m * n];
                let mut candidate_first = vec![0.0; m * n];
                let mut candidate_second = vec![0.0; m * n];
                let (matmul_baseline, matmul_candidate) = measure_thread_pair(
                    samples,
                    target,
                    candidate_threads,
                    || {
                        matmul_dispatch::matmul_dispatch_f32(&a, &b, &mut baseline_first, m, k, n)
                            .unwrap();
                        black_box(baseline_first[0]);
                    },
                    || {
                        matmul_dispatch::matmul_dispatch_f32(&a, &b, &mut candidate_first, m, k, n)
                            .unwrap();
                        black_box(candidate_first[0]);
                    },
                );
                assert_close(&baseline_first, &candidate_first, 1.0e-5);
                report_thread_pair(
                    m,
                    k,
                    n,
                    candidate_threads,
                    "matmul",
                    &matmul_baseline,
                    &matmul_candidate,
                );

                let (pipeline_baseline, pipeline_candidate) = measure_thread_pair(
                    samples,
                    target,
                    candidate_threads,
                    || {
                        run_preallocated_pipeline(
                            &a,
                            &b,
                            &residual,
                            &mut baseline_first,
                            &mut baseline_second,
                            m,
                            k,
                            n,
                        );
                    },
                    || {
                        run_preallocated_pipeline(
                            &a,
                            &b,
                            &residual,
                            &mut candidate_first,
                            &mut candidate_second,
                            m,
                            k,
                            n,
                        );
                    },
                );
                assert_close(&baseline_second, &candidate_second, 1.0e-5);
                report_thread_pair(
                    m,
                    k,
                    n,
                    candidate_threads,
                    "pipeline",
                    &pipeline_baseline,
                    &pipeline_candidate,
                );
            }
        }
    }
    bench_blas::configure(1);
}

#[allow(clippy::too_many_arguments)]
fn run_preallocated_pipeline(
    a: &[f32],
    b: &[f32],
    residual: &[f32],
    first: &mut [f32],
    second: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    matmul_dispatch::matmul_dispatch_f32(a, b, first, m, k, n).unwrap();
    add::add_f32(first, residual, second).unwrap();
    softmax::softmax_last_dim_f32(second, first, n).unwrap();
    transpose::transpose_f32(first, second, m, n).unwrap();
    black_box(second[0]);
}

fn assert_close(left: &[f32], right: &[f32], tolerance: f32) {
    assert_eq!(left.len(), right.len());
    for (&left, &right) in left.iter().zip(right) {
        let allowed = tolerance.max(left.abs() * tolerance);
        assert!((left - right).abs() <= allowed);
    }
}

fn report_thread_pair(
    m: usize,
    k: usize,
    n: usize,
    candidate_threads: usize,
    layer: &str,
    baseline: &Measurement,
    candidate: &Measurement,
) {
    println!(
        "blas_threads,{m}x{k}_x_{k}x{n},{candidate_threads},{layer},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.6},{:.6}",
        baseline.median.as_secs_f64() * 1e6,
        baseline.p95.as_secs_f64() * 1e6,
        baseline.p99.as_secs_f64() * 1e6,
        candidate.median.as_secs_f64() * 1e6,
        candidate.p95.as_secs_f64() * 1e6,
        candidate.p99.as_secs_f64() * 1e6,
        baseline.median.as_secs_f64() / candidate.median.as_secs_f64(),
        baseline.p95.as_secs_f64() / candidate.p95.as_secs_f64(),
        baseline.coefficient_of_variation,
        candidate.coefficient_of_variation,
    );
}

fn measure_thread_pair<F, G>(
    samples: usize,
    target: Duration,
    candidate_threads: usize,
    mut baseline: F,
    mut candidate: G,
) -> (Measurement, Measurement)
where
    F: FnMut(),
    G: FnMut(),
{
    bench_blas::configure(1);
    for _ in 0..5 {
        baseline();
    }
    let started = Instant::now();
    baseline();
    let baseline_once = started.elapsed().max(Duration::from_nanos(1));
    bench_blas::configure(candidate_threads);
    for _ in 0..5 {
        candidate();
    }
    let started = Instant::now();
    candidate();
    let candidate_once = started.elapsed().max(Duration::from_nanos(1));
    let slower = baseline_once.max(candidate_once);
    let iterations = (target.as_nanos().max(1) / slower.as_nanos()).clamp(1, 1_000_000) as usize;
    let mut baseline_timings = Vec::with_capacity(samples);
    let mut candidate_timings = Vec::with_capacity(samples);
    for sample in 0..samples {
        let mut run_baseline = || {
            bench_blas::configure(1);
            let started = Instant::now();
            for _ in 0..iterations {
                baseline();
            }
            baseline_timings.push(started.elapsed().div_f64(iterations as f64));
        };
        let mut run_candidate = || {
            bench_blas::configure(candidate_threads);
            let started = Instant::now();
            for _ in 0..iterations {
                candidate();
            }
            candidate_timings.push(started.elapsed().div_f64(iterations as f64));
        };
        if sample % 2 == 0 {
            run_baseline();
            run_candidate();
        } else {
            run_candidate();
            run_baseline();
        }
    }
    (
        summarize_measurements(baseline_timings, iterations),
        summarize_measurements(candidate_timings, iterations),
    )
}

fn benchmark_dispatch_overhead(samples: usize, target: Duration) {
    println!(
        "dispatch,case,selected_backend,select_median_ns,select_p95_ns,direct_median_us,direct_p95_us,dispatcher_median_us,dispatcher_p95_us,dispatcher_overhead_us,dispatcher_ratio"
    );
    for (m, k, n) in [
        (1, 768, 768),
        (2, 768, 768),
        (4, 768, 768),
        (2, 768, 3072),
        (2, 3072, 768),
        (128, 768, 768),
    ] {
        let backend = matmul_dispatch::select_backend(m, k, n, true);
        let selection = measure(samples, target, || {
            black_box(matmul_dispatch::select_backend(
                black_box(m),
                black_box(k),
                black_box(n),
                black_box(true),
            ));
        });
        let a = deterministic_values(m * k, 0.001);
        let b = deterministic_values(k * n, 0.002);
        let mut direct_output = vec![0.0; m * n];
        let mut dispatched_output = vec![0.0; m * n];
        let (direct, dispatched) = measure_pair(
            samples,
            target,
            || {
                run_selected_backend(backend, &a, &b, &mut direct_output, m, k, n).unwrap();
                black_box(direct_output[0]);
            },
            || {
                matmul_dispatch::matmul_dispatch_f32(&a, &b, &mut dispatched_output, m, k, n)
                    .unwrap();
                black_box(dispatched_output[0]);
            },
        );
        assert_eq!(direct_output, dispatched_output);
        let overhead = dispatched.median.saturating_sub(direct.median);
        println!(
            "dispatch,{m}x{k}_x_{k}x{n},{backend:?},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.6}",
            selection.median.as_secs_f64() * 1e9,
            selection.p95.as_secs_f64() * 1e9,
            direct.median.as_secs_f64() * 1e6,
            direct.p95.as_secs_f64() * 1e6,
            dispatched.median.as_secs_f64() * 1e6,
            dispatched.p95.as_secs_f64() * 1e6,
            overhead.as_secs_f64() * 1e6,
            dispatched.median.as_secs_f64() / direct.median.as_secs_f64(),
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn run_selected_backend(
    backend: matmul_dispatch::MatmulBackend,
    a: &[f32],
    b: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), matmul::MatmulError> {
    match backend {
        matmul_dispatch::MatmulBackend::Reference => matmul::matmul_f32(a, b, output, m, k, n),
        matmul_dispatch::MatmulBackend::CacheFriendly => {
            matmul::matmul_cache_friendly_f32(a, b, output, m, k, n)
        }
        matmul_dispatch::MatmulBackend::Tiled => matmul::matmul_tiled_f32(a, b, output, m, k, n),
        matmul_dispatch::MatmulBackend::Blas => blas::try_matmul_blas_f32(a, b, output, m, k, n)
            .unwrap_or_else(|| matmul::matmul_tiled_f32(a, b, output, m, k, n)),
    }
}

fn crossover_matmul_cases() -> Vec<(usize, usize, usize)> {
    const M_VALUES: [usize; 10] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
    const PROJECTIONS: [(usize, usize); 3] = [(768, 768), (768, 3072), (3072, 768)];
    let maximum_m = environment_usize("TRANSFORMER_BENCH_MAX_M", usize::MAX);
    let projection = env::var("TRANSFORMER_BENCH_PROJECTION").ok();

    PROJECTIONS
        .into_iter()
        .filter(|(k, n)| {
            projection
                .as_deref()
                .map_or(true, |expected| expected == format!("{k}x{n}"))
        })
        .flat_map(|(k, n)| {
            M_VALUES
                .into_iter()
                .filter(move |m| *m <= maximum_m)
                .map(move |m| (m, k, n))
        })
        .collect()
}

fn validate_blas_result(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    let mut expected = vec![0.0; m * n];
    let mut actual = vec![0.0; m * n];
    matmul::matmul_f32(a, b, &mut expected, m, k, n).unwrap();
    bench_blas::matmul_blas_f32(a, b, &mut actual, m, k, n).unwrap();

    for (&expected, &actual) in expected.iter().zip(&actual) {
        let tolerance = 1.0e-4_f32.max(expected.abs() * 1.0e-4);
        assert!(
            (expected - actual).abs() <= tolerance,
            "BLAS result mismatch"
        );
    }
}

type Matmul =
    fn(&[f32], &[f32], &mut [f32], usize, usize, usize) -> Result<(), matmul::MatmulError>;

fn benchmark_softmax(profile: Profile, samples: usize, target: Duration) {
    let sizes: &[usize] = match profile {
        Profile::Quick => &[128, 512, 2048],
        Profile::Transformer | Profile::Crossover => &[128, 512, 768, 2048, 8192, 32768],
    };

    for &length in sizes {
        let input = deterministic_values(length, 0.01);
        let mut output = vec![0.0; length];
        let measurement = measure(samples, target, || {
            softmax::softmax_f32(black_box(&input), black_box(&mut output)).unwrap();
            black_box(output[0]);
        });
        report(
            "softmax_f32",
            &format!("n={length}"),
            measurement,
            Throughput::Gigabytes(8.0 * length as f64),
        );
    }

    let input = deterministic_values(128 * 768, 0.01);
    let mut rank_one_output = vec![0.0; input.len()];
    let rank_one = measure(samples, target, || {
        let input = black_box(&input);
        let output = black_box(&mut rank_one_output);
        for (input_row, output_row) in input.chunks_exact(768).zip(output.chunks_exact_mut(768)) {
            softmax::softmax_f32(input_row, output_row).unwrap();
        }
        black_box(rank_one_output[0]);
    });
    report(
        "softmax_f32_x128",
        "128x768_last_dim",
        rank_one,
        Throughput::Gigabytes(8.0 * input.len() as f64),
    );

    let mut rank_two_output = vec![0.0; input.len()];
    let rank_two = measure(samples, target, || {
        softmax::softmax_last_dim_f32(black_box(&input), black_box(&mut rank_two_output), 768)
            .unwrap();
        black_box(rank_two_output[0]);
    });
    report(
        "softmax_last_dim_f32",
        "128x768_last_dim",
        rank_two,
        Throughput::Gigabytes(8.0 * input.len() as f64),
    );
    benchmark_combined_softmax_pass(samples, target);
}

fn softmax_combined_validation_max_f32(
    input: &[f32],
    output: &mut [f32],
) -> Result<(), softmax::SoftmaxError> {
    if input.is_empty() {
        return Err(softmax::SoftmaxError::EmptyInput);
    }
    if input.len() != output.len() {
        return Err(softmax::SoftmaxError::LengthMismatch);
    }
    let mut maximum = input[0];
    if !maximum.is_finite() {
        return Err(softmax::SoftmaxError::NonFiniteInput);
    }
    for &value in &input[1..] {
        if !value.is_finite() {
            return Err(softmax::SoftmaxError::NonFiniteInput);
        }
        maximum = maximum.max(value);
    }
    let mut sum = 0.0;
    for (destination, &value) in output.iter_mut().zip(input) {
        let exponential = (value - maximum).exp();
        *destination = exponential;
        sum += exponential;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err(softmax::SoftmaxError::InvalidNormalization);
    }
    for value in output {
        *value /= sum;
    }
    Ok(())
}

fn softmax_last_dim_combined_f32(
    input: &[f32],
    output: &mut [f32],
    last_dim: usize,
) -> Result<(), softmax::SoftmaxError> {
    if input.is_empty() || last_dim == 0 {
        return Err(softmax::SoftmaxError::EmptyInput);
    }
    if input.len() != output.len() || input.len() % last_dim != 0 {
        return Err(softmax::SoftmaxError::LengthMismatch);
    }
    for (input_row, output_row) in input
        .chunks_exact(last_dim)
        .zip(output.chunks_exact_mut(last_dim))
    {
        softmax_combined_validation_max_f32(input_row, output_row)?;
    }
    Ok(())
}

fn benchmark_combined_softmax_pass(samples: usize, target: Duration) {
    let input = deterministic_values(128 * 768, 0.01);
    let mut expected = vec![0.0; input.len()];
    let mut actual = vec![0.0; input.len()];
    softmax::softmax_last_dim_f32(&input, &mut expected, 768).unwrap();
    softmax_last_dim_combined_f32(&input, &mut actual, 768).unwrap();
    assert_eq!(actual, expected);
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            softmax_combined_validation_max_f32(&[0.0, invalid], &mut [0.0; 2]),
            softmax::softmax_f32(&[0.0, invalid], &mut [0.0; 2])
        );
    }
    let (baseline, candidate) = measure_pair(
        samples,
        target,
        || {
            softmax::softmax_last_dim_f32(&input, &mut expected, 768).unwrap();
            black_box(expected[0]);
        },
        || {
            softmax_last_dim_combined_f32(&input, &mut actual, 768).unwrap();
            black_box(actual[0]);
        },
    );
    println!(
        "softmax_combined,128x768,kernel,{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
        baseline.median.as_secs_f64() * 1e6,
        baseline.p95.as_secs_f64() * 1e6,
        candidate.median.as_secs_f64() * 1e6,
        candidate.p95.as_secs_f64() * 1e6,
        baseline.median.as_secs_f64() / candidate.median.as_secs_f64(),
        baseline.p95.as_secs_f64() / candidate.p95.as_secs_f64(),
    );

    let (m, k, n) = (128, 768, 768);
    let a = deterministic_values(m * k, 0.001);
    let b = deterministic_values(k * n, 0.002);
    let residual = deterministic_values(m * n, 0.003);
    let (pipeline_baseline, pipeline_candidate) = measure_pair(
        samples,
        target,
        || {
            black_box(run_allocating_pipeline(&a, &b, &residual, m, k, n));
        },
        || {
            black_box(run_pipeline_combined_softmax(&a, &b, &residual, m, k, n));
        },
    );
    assert_eq!(
        run_pipeline_combined_softmax(&a, &b, &residual, m, k, n).values,
        run_allocating_pipeline(&a, &b, &residual, m, k, n).values
    );
    println!(
        "softmax_combined,128x768,pipeline,{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
        pipeline_baseline.median.as_secs_f64() * 1e6,
        pipeline_baseline.p95.as_secs_f64() * 1e6,
        pipeline_candidate.median.as_secs_f64() * 1e6,
        pipeline_candidate.p95.as_secs_f64() * 1e6,
        pipeline_baseline.median.as_secs_f64() / pipeline_candidate.median.as_secs_f64(),
        pipeline_baseline.p95.as_secs_f64() / pipeline_candidate.p95.as_secs_f64(),
    );
}

fn softmax_vector_exp_f32(input: &[f32], output: &mut [f32]) -> Result<(), softmax::SoftmaxError> {
    if input.is_empty() {
        return Err(softmax::SoftmaxError::EmptyInput);
    }
    if input.len() != output.len() {
        return Err(softmax::SoftmaxError::LengthMismatch);
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(softmax::SoftmaxError::NonFiniteInput);
    }
    let maximum = input[1..].iter().copied().fold(input[0], f32::max);
    let mut sum = 0.0;
    let mut processed = 0;
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        // SAFETY: Runtime feature detection covers every required instruction,
        // and the helper reads/writes exactly eight valid elements per chunk.
        processed = unsafe { vector_exp_chunks(input, output, maximum, &mut sum) };
    }
    for index in processed..input.len() {
        let exponential = (input[index] - maximum).exp();
        output[index] = exponential;
        sum += exponential;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err(softmax::SoftmaxError::InvalidNormalization);
    }
    for value in output {
        *value /= sum;
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vector_exp_chunks(
    input: &[f32],
    output: &mut [f32],
    maximum: f32,
    sum: &mut f32,
) -> usize {
    use std::arch::x86_64::*;

    let processed = input.len() / 8 * 8;
    let maximum = _mm256_set1_ps(maximum);
    for index in (0..processed).step_by(8) {
        // SAFETY: `processed` is rounded down to complete eight-float chunks.
        let values = unsafe { _mm256_loadu_ps(input.as_ptr().add(index)) };
        // SAFETY: This function has the same enabled target features.
        let exponentials = unsafe { exp256_ps(_mm256_sub_ps(values, maximum)) };
        // SAFETY: The output length equals the input length and this is a full chunk.
        unsafe { _mm256_storeu_ps(output.as_mut_ptr().add(index), exponentials) };
        for value in &output[index..index + 8] {
            *sum += *value;
        }
    }
    processed
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp256_ps(input: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    // SAFETY: The caller guarantees AVX2 and FMA support.
    unsafe {
        let input = _mm256_max_ps(
            _mm256_set1_ps(-88.376_26),
            _mm256_min_ps(input, _mm256_set1_ps(88.376_26)),
        );
        let fx = _mm256_floor_ps(_mm256_fmadd_ps(
            input,
            _mm256_set1_ps(std::f32::consts::LOG2_E),
            _mm256_set1_ps(0.5),
        ));
        let reduced = _mm256_sub_ps(
            _mm256_sub_ps(input, _mm256_mul_ps(fx, _mm256_set1_ps(0.693_359_4))),
            _mm256_mul_ps(fx, _mm256_set1_ps(-2.121_944_4e-4)),
        );
        let squared = _mm256_mul_ps(reduced, reduced);
        let mut polynomial = _mm256_set1_ps(1.987_569_1e-4);
        for coefficient in [
            1.398_2e-3,
            8.333_452e-3,
            4.166_579_6e-2,
            1.666_666_6e-1,
            5.0e-1,
        ] {
            polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(coefficient));
        }
        polynomial = _mm256_fmadd_ps(polynomial, squared, reduced);
        polynomial = _mm256_add_ps(polynomial, _mm256_set1_ps(1.0));
        let exponent = _mm256_slli_epi32(
            _mm256_add_epi32(_mm256_cvttps_epi32(fx), _mm256_set1_epi32(127)),
            23,
        );
        _mm256_mul_ps(polynomial, _mm256_castsi256_ps(exponent))
    }
}

fn softmax_last_dim_vector_exp_f32(
    input: &[f32],
    output: &mut [f32],
    last_dim: usize,
) -> Result<(), softmax::SoftmaxError> {
    if input.is_empty() || last_dim == 0 {
        return Err(softmax::SoftmaxError::EmptyInput);
    }
    if input.len() != output.len() || input.len() % last_dim != 0 {
        return Err(softmax::SoftmaxError::LengthMismatch);
    }
    for (input_row, output_row) in input
        .chunks_exact(last_dim)
        .zip(output.chunks_exact_mut(last_dim))
    {
        softmax_vector_exp_f32(input_row, output_row)?;
    }
    Ok(())
}

fn benchmark_vector_exp_softmax(samples: usize, target: Duration) {
    #[cfg(target_arch = "x86_64")]
    println!(
        "vector_exp_cpu,avx2={},fma={}",
        std::is_x86_feature_detected!("avx2"),
        std::is_x86_feature_detected!("fma")
    );
    println!("vector_exp,rows,last_dim,layer,baseline_median_us,baseline_p95_us,baseline_p99_us,candidate_median_us,candidate_p95_us,candidate_p99_us,median_speedup,p95_speedup,max_abs,max_rel,max_ulp,max_sum_error");
    for last_dim in [128, 768, 2048, 3072] {
        for rows in [1, 2, 8, 32, 128] {
            let input = deterministic_values(rows * last_dim, 0.01);
            let mut baseline_output = vec![0.0; input.len()];
            let mut candidate_output = vec![0.0; input.len()];
            softmax::softmax_last_dim_f32(&input, &mut baseline_output, last_dim).unwrap();
            softmax_last_dim_vector_exp_f32(&input, &mut candidate_output, last_dim).unwrap();
            let errors = softmax_errors(&baseline_output, &candidate_output, last_dim);
            let (baseline, candidate) = measure_pair(
                samples,
                target,
                || {
                    softmax::softmax_last_dim_f32(&input, &mut baseline_output, last_dim).unwrap();
                    black_box(baseline_output[0]);
                },
                || {
                    softmax_last_dim_vector_exp_f32(&input, &mut candidate_output, last_dim)
                        .unwrap();
                    black_box(candidate_output[0]);
                },
            );
            report_vector_exp(rows, last_dim, "kernel", &baseline, &candidate, errors);
        }
    }

    for (k, n) in [(768, 768), (768, 3072), (3072, 768)] {
        let m = 128;
        let a = deterministic_values(m * k, 0.001);
        let b = deterministic_values(k * n, 0.002);
        let residual = deterministic_values(m * n, 0.003);
        let (baseline, candidate) = measure_pair(
            samples,
            target,
            || {
                black_box(run_allocating_pipeline(&a, &b, &residual, m, k, n));
            },
            || {
                black_box(run_pipeline_vector_exp_softmax(&a, &b, &residual, m, k, n));
            },
        );
        let expected = run_allocating_pipeline(&a, &b, &residual, m, k, n);
        let actual = run_pipeline_vector_exp_softmax(&a, &b, &residual, m, k, n);
        let pointwise = pointwise_errors(&expected.values, &actual.values);
        let errors = (pointwise.0, pointwise.1, pointwise.2, 0.0);
        report_vector_exp(
            m,
            n,
            &format!("pipeline_k{k}"),
            &baseline,
            &candidate,
            errors,
        );
    }

    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            softmax_vector_exp_f32(&[0.0, invalid], &mut [0.0; 2]),
            softmax::softmax_f32(&[0.0, invalid], &mut [0.0; 2])
        );
    }
}

fn softmax_errors(expected: &[f32], actual: &[f32], last_dim: usize) -> (f32, f32, u32, f32) {
    let pointwise = pointwise_errors(expected, actual);
    let maximum_sum_error = actual
        .chunks_exact(last_dim)
        .map(|row| (row.iter().sum::<f32>() - 1.0).abs())
        .fold(0.0_f32, f32::max);
    (pointwise.0, pointwise.1, pointwise.2, maximum_sum_error)
}

fn pointwise_errors(expected: &[f32], actual: &[f32]) -> (f32, f32, u32) {
    let mut maximum_absolute = 0.0_f32;
    let mut maximum_relative = 0.0_f32;
    let mut maximum_ulp = 0_u32;
    for (&expected, &actual) in expected.iter().zip(actual) {
        let absolute = (expected - actual).abs();
        maximum_absolute = maximum_absolute.max(absolute);
        if expected != 0.0 {
            maximum_relative = maximum_relative.max(absolute / expected.abs());
        }
        maximum_ulp = maximum_ulp.max(expected.to_bits().abs_diff(actual.to_bits()));
    }
    (maximum_absolute, maximum_relative, maximum_ulp)
}

fn report_vector_exp(
    rows: usize,
    last_dim: usize,
    layer: &str,
    baseline: &Measurement,
    candidate: &Measurement,
    errors: (f32, f32, u32, f32),
) {
    println!(
        "vector_exp,{rows},{last_dim},{layer},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.9},{:.9},{},{:.9}",
        baseline.median.as_secs_f64() * 1e6,
        baseline.p95.as_secs_f64() * 1e6,
        baseline.p99.as_secs_f64() * 1e6,
        candidate.median.as_secs_f64() * 1e6,
        candidate.p95.as_secs_f64() * 1e6,
        candidate.p99.as_secs_f64() * 1e6,
        baseline.median.as_secs_f64() / candidate.median.as_secs_f64(),
        baseline.p95.as_secs_f64() / candidate.p95.as_secs_f64(),
        errors.0,
        errors.1,
        errors.2,
        errors.3,
    );
}

fn run_pipeline_vector_exp_softmax(
    a: &[f32],
    b: &[f32],
    residual: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Box<OwnedBenchTensor> {
    let mut multiplied = vec![0.0; m * n];
    let mut added = vec![0.0; m * n];
    let mut normalized = vec![0.0; m * n];
    let mut transposed = vec![0.0; m * n];
    matmul_dispatch::matmul_dispatch_f32(a, b, &mut multiplied, m, k, n).unwrap();
    add::add_f32(&multiplied, residual, &mut added).unwrap();
    softmax_last_dim_vector_exp_f32(&added, &mut normalized, n).unwrap();
    transpose::transpose_f32(&normalized, &mut transposed, m, n).unwrap();
    Box::new(OwnedBenchTensor {
        values: transposed,
        rows: n,
        columns: m,
    })
}

fn run_pipeline_combined_softmax(
    a: &[f32],
    b: &[f32],
    residual: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Box<OwnedBenchTensor> {
    let mut multiplied = vec![0.0; m * n];
    let mut added = vec![0.0; m * n];
    let mut normalized = vec![0.0; m * n];
    let mut transposed = vec![0.0; m * n];
    matmul_dispatch::matmul_dispatch_f32(a, b, &mut multiplied, m, k, n).unwrap();
    add::add_f32(&multiplied, residual, &mut added).unwrap();
    softmax_last_dim_combined_f32(&added, &mut normalized, n).unwrap();
    transpose::transpose_f32(&normalized, &mut transposed, m, n).unwrap();
    Box::new(OwnedBenchTensor {
        values: transposed,
        rows: n,
        columns: m,
    })
}

fn benchmark_add(profile: Profile, samples: usize, target: Duration) {
    let sizes: &[usize] = match profile {
        Profile::Quick => &[768, 98_304, 1_048_576],
        Profile::Transformer | Profile::Crossover => {
            &[768, 98_304, 1_048_576, 4_194_304, 16_777_216]
        }
    };

    for &length in sizes {
        let a = deterministic_values(length, 0.001);
        let b = deterministic_values(length, 0.002);
        let mut output = vec![0.0; length];
        let measurement = measure(samples, target, || {
            add::add_f32(black_box(&a), black_box(&b), black_box(&mut output)).unwrap();
            black_box(output[0]);
        });
        report(
            "add_f32",
            &format!("n={length}"),
            measurement,
            Throughput::Gigabytes(12.0 * length as f64),
        );
    }
}

fn benchmark_transpose(profile: Profile, samples: usize, target: Duration) {
    let cases: &[(usize, usize)] = match profile {
        Profile::Quick => &[(128, 768), (768, 768)],
        Profile::Transformer | Profile::Crossover => {
            &[(128, 768), (768, 768), (768, 3072), (3072, 768)]
        }
    };

    validate_transpose_candidates();
    println!("transpose_a_b,shape,variant,baseline_median_us,baseline_p95_us,baseline_p99_us,candidate_median_us,candidate_p95_us,candidate_p99_us,median_speedup,p95_speedup");
    for &(rows, columns) in cases {
        let input = deterministic_values(rows * columns, 0.001);
        let mut output = vec![0.0; rows * columns];
        let measurement = measure(samples, target, || {
            transpose::transpose_f32(black_box(&input), black_box(&mut output), rows, columns)
                .unwrap();
            black_box(output[0]);
        });
        report(
            "transpose_f32",
            &format!("{rows}x{columns}"),
            measurement,
            Throughput::Gigabytes(8.0 * rows as f64 * columns as f64),
        );

        for tile in [4, 8, 16, 32, 64, 128] {
            let mut baseline_output = vec![0.0; rows * columns];
            let mut candidate_output = vec![0.0; rows * columns];
            transpose::transpose_f32(&input, &mut baseline_output, rows, columns).unwrap();
            transpose_tiled_f32(&input, &mut candidate_output, rows, columns, tile);
            assert_eq!(candidate_output, baseline_output);
            let (baseline, candidate) = measure_pair(
                samples,
                target,
                || {
                    transpose::transpose_f32(&input, &mut baseline_output, rows, columns).unwrap();
                    black_box(baseline_output[0]);
                },
                || {
                    transpose_tiled_f32(&input, &mut candidate_output, rows, columns, tile);
                    black_box(candidate_output[0]);
                },
            );
            println!(
                "transpose_a_b,{rows}x{columns},tile_{tile},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
                baseline.median.as_secs_f64() * 1e6,
                baseline.p95.as_secs_f64() * 1e6,
                baseline.p99.as_secs_f64() * 1e6,
                candidate.median.as_secs_f64() * 1e6,
                candidate.p95.as_secs_f64() * 1e6,
                candidate.p99.as_secs_f64() * 1e6,
                baseline.median.as_secs_f64() / candidate.median.as_secs_f64(),
                baseline.p95.as_secs_f64() / candidate.p95.as_secs_f64(),
            );
        }
    }

    benchmark_transpose_pipeline(samples, target);
    benchmark_transpose_tensor_model(samples, target);
    benchmark_transpose_selection(samples, target);
}

fn validate_transpose_candidates() {
    for (rows, columns) in [(1, 1), (2, 3), (3, 5), (7, 17), (31, 33), (128, 768)] {
        let input: Vec<f32> = (0..rows * columns)
            .map(|index| match index % 5 {
                0 => 0.0,
                1 => -(index as f32),
                2 => 1.0e20,
                3 => 1.0e-20,
                _ => index as f32,
            })
            .collect();
        let mut expected = vec![0.0; input.len()];
        transpose::transpose_f32(&input, &mut expected, rows, columns).unwrap();
        for tile in [4, 8, 16, 32, 64, 128] {
            let mut actual = vec![f32::NAN; input.len()];
            transpose_tiled_f32(&input, &mut actual, rows, columns, tile);
            assert_eq!(
                actual, expected,
                "transpose parity failed for {rows}x{columns}, tile={tile}"
            );
        }
    }
}

fn transpose_tiled_f32(
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    columns: usize,
    tile: usize,
) {
    for row_start in (0..rows).step_by(tile) {
        let row_end = (row_start + tile).min(rows);
        for column_start in (0..columns).step_by(tile) {
            let column_end = (column_start + tile).min(columns);
            for row in row_start..row_end {
                for column in column_start..column_end {
                    output[column * rows + row] = input[row * columns + column];
                }
            }
        }
    }
}

fn benchmark_transpose_pipeline(samples: usize, target: Duration) {
    let (m, k, n) = (128, 768, 768);
    let a = deterministic_values(m * k, 0.001);
    let b = deterministic_values(k * n, 0.002);
    let residual = deterministic_values(m * n, 0.003);
    println!(
        "transpose_pipeline,variant,baseline_median_us,baseline_p95_us,baseline_p99_us,candidate_median_us,candidate_p95_us,candidate_p99_us,median_speedup,p95_speedup"
    );
    for tile in [4, 8, 16, 32, 64, 128] {
        let (baseline, candidate) = measure_pair(
            samples,
            target,
            || {
                black_box(run_allocating_pipeline(&a, &b, &residual, m, k, n));
            },
            || {
                black_box(run_allocating_pipeline_tiled_transpose(
                    &a, &b, &residual, m, k, n, tile,
                ));
            },
        );
        let expected = run_allocating_pipeline(&a, &b, &residual, m, k, n);
        let actual = run_allocating_pipeline_tiled_transpose(&a, &b, &residual, m, k, n, tile);
        assert_eq!(actual.values, expected.values);
        println!(
            "transpose_pipeline,tile_{tile},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
            baseline.median.as_secs_f64() * 1e6,
            baseline.p95.as_secs_f64() * 1e6,
            baseline.p99.as_secs_f64() * 1e6,
            candidate.median.as_secs_f64() * 1e6,
            candidate.p95.as_secs_f64() * 1e6,
            candidate.p99.as_secs_f64() * 1e6,
            baseline.median.as_secs_f64() / candidate.median.as_secs_f64(),
            baseline.p95.as_secs_f64() / candidate.p95.as_secs_f64(),
        );
    }
}

fn benchmark_transpose_tensor_model(samples: usize, target: Duration) {
    let (m, k, n) = (128, 768, 768);
    let a = tensor::Tensor::from_vec(
        deterministic_values(m * k, 0.001),
        tensor::Shape::new(vec![m, k]).unwrap(),
    )
    .unwrap();
    let b = tensor::Tensor::from_vec(
        deterministic_values(k * n, 0.002),
        tensor::Shape::new(vec![k, n]).unwrap(),
    )
    .unwrap();
    let residual = tensor::Tensor::from_vec(
        deterministic_values(m * n, 0.003),
        tensor::Shape::new(vec![m, n]).unwrap(),
    )
    .unwrap();
    println!("transpose_tensor_model,variant,baseline_median_us,baseline_p95_us,candidate_median_us,candidate_p95_us,median_speedup,p95_speedup");
    for tile in [4, 8, 16, 32, 64, 128] {
        let (baseline, candidate) = measure_pair(
            samples,
            target,
            || {
                black_box(run_tensor_pipeline(&a, &b, &residual, None));
            },
            || {
                black_box(run_tensor_pipeline(&a, &b, &residual, Some(tile)));
            },
        );
        assert_eq!(
            run_tensor_pipeline(&a, &b, &residual, None).as_slice(),
            run_tensor_pipeline(&a, &b, &residual, Some(tile)).as_slice()
        );
        println!(
            "transpose_tensor_model,tile_{tile},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
            baseline.median.as_secs_f64() * 1e6,
            baseline.p95.as_secs_f64() * 1e6,
            candidate.median.as_secs_f64() * 1e6,
            candidate.p95.as_secs_f64() * 1e6,
            baseline.median.as_secs_f64() / candidate.median.as_secs_f64(),
            baseline.p95.as_secs_f64() / candidate.p95.as_secs_f64(),
        );
    }
}

fn run_tensor_pipeline(
    a: &tensor::Tensor,
    b: &tensor::Tensor,
    residual: &tensor::Tensor,
    transpose_tile: Option<usize>,
) -> tensor::Tensor {
    let (m, k, n) = (
        a.shape().as_slice()[0],
        a.shape().as_slice()[1],
        b.shape().as_slice()[1],
    );
    let shape = tensor::Shape::new(vec![m, n]).unwrap();
    let mut multiplied = vec![0.0; shape.numel()];
    matmul_dispatch::matmul_dispatch_f32(a.as_slice(), b.as_slice(), &mut multiplied, m, k, n)
        .unwrap();
    let multiplied = tensor::Tensor::from_vec(multiplied, shape.clone()).unwrap();
    let mut added = vec![0.0; shape.numel()];
    add::add_f32(multiplied.as_slice(), residual.as_slice(), &mut added).unwrap();
    let added = tensor::Tensor::from_vec(added, shape.clone()).unwrap();
    let mut normalized = vec![0.0; shape.numel()];
    softmax::softmax_last_dim_f32(added.as_slice(), &mut normalized, n).unwrap();
    let normalized = tensor::Tensor::from_vec(normalized, shape).unwrap();
    let output_shape = tensor::Shape::new(vec![n, m]).unwrap();
    let mut output = vec![0.0; output_shape.numel()];
    if let Some(tile) = transpose_tile {
        transpose_tiled_f32(normalized.as_slice(), &mut output, m, n, tile);
    } else {
        transpose::transpose_f32(normalized.as_slice(), &mut output, m, n).unwrap();
    }
    tensor::Tensor::from_vec(output, output_shape).unwrap()
}

fn select_transpose_tile(rows: usize, columns: usize) -> usize {
    if rows > columns {
        8
    } else {
        16
    }
}

fn benchmark_transpose_selection(samples: usize, target: Duration) {
    for (rows, columns) in [(128, 768), (768, 768), (768, 3072), (3072, 768)] {
        let measurement = measure(samples, target, || {
            black_box(select_transpose_tile(black_box(rows), black_box(columns)));
        });
        println!(
            "transpose_selection,{rows}x{columns},tile_{},{:.3},{:.3},{:.3}",
            select_transpose_tile(rows, columns),
            measurement.median.as_secs_f64() * 1e9,
            measurement.p95.as_secs_f64() * 1e9,
            measurement.p99.as_secs_f64() * 1e9,
        );
    }
}

fn run_allocating_pipeline_tiled_transpose(
    a: &[f32],
    b: &[f32],
    residual: &[f32],
    m: usize,
    k: usize,
    n: usize,
    tile: usize,
) -> Box<OwnedBenchTensor> {
    let mut current = vec![0.0; m * n];
    let mut scratch = vec![0.0; m * n];
    matmul_dispatch::matmul_dispatch_f32(a, b, &mut current, m, k, n).unwrap();
    add::add_f32(&current, residual, &mut scratch).unwrap();
    softmax::softmax_last_dim_f32(&scratch, &mut current, n).unwrap();
    transpose_tiled_f32(&current, &mut scratch, m, n, tile);
    Box::new(OwnedBenchTensor {
        values: scratch,
        rows: n,
        columns: m,
    })
}

fn measure<F>(samples: usize, target: Duration, mut operation: F) -> Measurement
where
    F: FnMut(),
{
    for _ in 0..3 {
        operation();
    }

    let calibration_start = Instant::now();
    operation();
    let one_iteration = calibration_start.elapsed().max(Duration::from_nanos(1));
    let target_nanos = target.as_nanos().max(1);
    let iterations = (target_nanos / one_iteration.as_nanos()).clamp(1, 1_000_000) as usize;

    let mut timings = Vec::with_capacity(samples);
    for _ in 0..samples {
        let started = Instant::now();
        for _ in 0..iterations {
            operation();
        }
        timings.push(started.elapsed().div_f64(iterations as f64));
    }
    timings.sort_unstable();

    Measurement {
        median: timings[timings.len() / 2],
        p95: timings[((timings.len() - 1) * 95).div_ceil(100)],
        p99: timings[((timings.len() - 1) * 99).div_ceil(100)],
        coefficient_of_variation: coefficient_of_variation(&timings),
        iterations,
    }
}

fn measure_pair<F, G>(
    samples: usize,
    target: Duration,
    mut baseline: F,
    mut candidate: G,
) -> (Measurement, Measurement)
where
    F: FnMut(),
    G: FnMut(),
{
    for _ in 0..3 {
        baseline();
        candidate();
    }

    let started = Instant::now();
    baseline();
    let baseline_once = started.elapsed().max(Duration::from_nanos(1));
    let started = Instant::now();
    candidate();
    let candidate_once = started.elapsed().max(Duration::from_nanos(1));
    let slower = baseline_once.max(candidate_once);
    let iterations = (target.as_nanos().max(1) / slower.as_nanos()).clamp(1, 1_000_000) as usize;

    let mut baseline_timings = Vec::with_capacity(samples);
    let mut candidate_timings = Vec::with_capacity(samples);
    for sample in 0..samples {
        let mut measure_baseline = || {
            let started = Instant::now();
            for _ in 0..iterations {
                baseline();
            }
            baseline_timings.push(started.elapsed().div_f64(iterations as f64));
        };
        let mut measure_candidate = || {
            let started = Instant::now();
            for _ in 0..iterations {
                candidate();
            }
            candidate_timings.push(started.elapsed().div_f64(iterations as f64));
        };

        if sample % 2 == 0 {
            measure_baseline();
            measure_candidate();
        } else {
            measure_candidate();
            measure_baseline();
        }
    }
    baseline_timings.sort_unstable();
    candidate_timings.sort_unstable();

    let summarize = |timings: &[Duration]| Measurement {
        median: timings[timings.len() / 2],
        p95: timings[((timings.len() - 1) * 95).div_ceil(100)],
        p99: timings[((timings.len() - 1) * 99).div_ceil(100)],
        coefficient_of_variation: coefficient_of_variation(timings),
        iterations,
    };
    (summarize(&baseline_timings), summarize(&candidate_timings))
}

fn coefficient_of_variation(timings: &[Duration]) -> f64 {
    let mean = timings.iter().map(Duration::as_secs_f64).sum::<f64>() / timings.len() as f64;
    if mean == 0.0 {
        return 0.0;
    }
    let variance = timings
        .iter()
        .map(|timing| {
            let difference = timing.as_secs_f64() - mean;
            difference * difference
        })
        .sum::<f64>()
        / timings.len() as f64;
    variance.sqrt() / mean
}

fn summarize_measurements(mut timings: Vec<Duration>, iterations: usize) -> Measurement {
    timings.sort_unstable();
    Measurement {
        median: timings[timings.len() / 2],
        p95: timings[((timings.len() - 1) * 95).div_ceil(100)],
        p99: timings[((timings.len() - 1) * 99).div_ceil(100)],
        coefficient_of_variation: coefficient_of_variation(&timings),
        iterations,
    }
}

fn report(kernel: &str, case: &str, measurement: Measurement, throughput: Throughput) {
    let seconds = measurement.median.as_secs_f64();
    let Throughput::Gigabytes(bytes) = throughput;
    let metric = "GB/s";
    let value = bytes / seconds / 1e9;

    println!(
        "{kernel},{BASELINE},{case},{:.3},{:.3},{},{metric},{value:.3},,,",
        measurement.median.as_secs_f64() * 1e6,
        measurement.p95.as_secs_f64() * 1e6,
        measurement.iterations,
    );
}

fn report_matmul(
    variant: &str,
    case: &str,
    measurement: Measurement,
    operations: f64,
    scalar: Duration,
    tiled: Duration,
    winner: &str,
) {
    let gflops = operations / measurement.median.as_secs_f64() / 1e9;
    let speedup_scalar = scalar.as_secs_f64() / measurement.median.as_secs_f64();
    let speedup_tiled = tiled.as_secs_f64() / measurement.median.as_secs_f64();

    println!(
        "matmul_f32,{variant},{case},{:.3},{:.3},{},GFLOP/s,{gflops:.3},{speedup_scalar:.3},{speedup_tiled:.3},{winner}",
        measurement.median.as_secs_f64() * 1e6,
        measurement.p95.as_secs_f64() * 1e6,
        measurement.iterations,
    );
}

fn deterministic_values(length: usize, scale: f32) -> Vec<f32> {
    (0..length)
        .map(|index| ((index % 251) as f32 - 125.0) * scale)
        .collect()
}

fn matches_filter(filter: &str, kernel: &str) -> bool {
    filter.is_empty() || kernel.contains(filter)
}

fn environment_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .map(|value| value.parse().unwrap_or_else(|_| panic!("invalid {name}")))
        .unwrap_or(default)
}

fn environment_u64(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .map(|value| value.parse().unwrap_or_else(|_| panic!("invalid {name}")))
        .unwrap_or(default)
}
