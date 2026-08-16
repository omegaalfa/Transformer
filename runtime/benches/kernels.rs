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
        }
        if matches_filter(&filter, "fusion") {
            benchmark_matmul_add_fusion(samples, target);
        }
        if matches_filter(&filter, "lifecycle") {
            benchmark_storage_reuse(samples, target);
        }
    }
    if matches_filter(&filter, "softmax") {
        benchmark_softmax(profile, samples, target);
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

fn crossover_matmul_cases() -> Vec<(usize, usize, usize)> {
    const M_VALUES: [usize; 10] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
    const PROJECTIONS: [(usize, usize); 3] = [(768, 768), (768, 3072), (3072, 768)];

    PROJECTIONS
        .into_iter()
        .flat_map(|(k, n)| M_VALUES.into_iter().map(move |m| (m, k, n)))
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
    }
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
        iterations,
    };
    (summarize(&baseline_timings), summarize(&candidate_timings))
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
