use std::env;
use std::hint::black_box;
use std::time::{Duration, Instant};

#[allow(dead_code, unused_imports)]
#[path = "../src/kernels/add.rs"]
mod add;
#[allow(dead_code)]
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
    pub(crate) use crate::{add, matmul, matmul_dispatch, softmax, transpose};
}

#[allow(dead_code)]
#[path = "../src/graph.rs"]
mod graph;

use graph::{
    execute, plan_storage, simulate, ExecutionPlan, ExternalInputs, ExternalTensor, Graph,
    GraphBuilder, GraphDType, Layout, Operation, ValueId,
};

const DEFAULT_SAMPLES: usize = 25;
const DEFAULT_TARGET_US: u64 = 1_000;

struct Measurement {
    median: Duration,
    p95: Duration,
    iterations: usize,
}

fn main() {
    let blas = bench_blas::configure(1);
    let samples = environment_usize("TRANSFORMER_GRAPH_BENCH_SAMPLES", DEFAULT_SAMPLES).max(3);
    let target = Duration::from_micros(environment_u64(
        "TRANSFORMER_GRAPH_BENCH_TARGET_US",
        DEFAULT_TARGET_US,
    ));
    println!(
        "Graph planner benchmarks samples={samples} target_us={}",
        target.as_micros()
    );
    println!(
        "blas_backend=openblas parallel={} threads={} core={}",
        blas.parallel, blas.threads, blas.core
    );
    println!("case,stage,nodes,median_us,p95_us,iterations,slots,allocations,reuses,peak_bytes");

    benchmark_case("transformer_4", GraphCase::Transformer, samples, target);
    benchmark_case("linear_10", GraphCase::Linear(10), samples, target);
    benchmark_case("branching", GraphCase::Branching, samples, target);
    benchmark_case("linear_100", GraphCase::Linear(100), samples, target);
    benchmark_case("linear_1000", GraphCase::Linear(1000), samples, target);
    benchmark_case("linear_10000", GraphCase::Linear(10_000), samples, target);
    benchmark_numeric_pipeline(samples, Duration::from_millis(20));
}

fn benchmark_numeric_pipeline(samples: usize, target: Duration) {
    let (m, k, n) = (128, 768, 768);
    let a = deterministic_values(m * k, 0.001);
    let b = deterministic_values(k * n, 0.002);
    let residual = deterministic_values(m * n, 0.003);
    let (graph, ids) = build_transformer_shape(m, k, n);
    let storage = plan_storage(&graph).unwrap();
    let execution_plan = ExecutionPlan::new(&graph, &storage).unwrap();

    let plan_build = measure(samples, target, || {
        let (mut graph, _) = build_transformer_unanalyzed(m, k, n);
        graph.analyze_lifetimes();
        let storage = plan_storage(&graph).unwrap();
        black_box(ExecutionPlan::new(&graph, &storage).unwrap());
    });
    let plan_validate = measure(samples, target, || {
        execution_plan.validate(&graph, &storage).unwrap();
        black_box(());
    });
    let (baseline, planned) = measure_pair(
        samples,
        target,
        || black_box(run_allocating_pipeline(&a, &b, &residual, m, k, n)),
        || {
            black_box(run_graph_pipeline(
                &graph,
                &storage,
                &execution_plan,
                ids,
                &a,
                &b,
                &residual,
            ))
        },
    );
    let result = run_graph_pipeline(&graph, &storage, &execution_plan, ids, &a, &b, &residual);
    let (expected, baseline_stages) = run_allocating_pipeline_timed(&a, &b, &residual, m, k, n);
    assert_eq!(result.outputs[0].data, expected);

    println!("numeric,variant,median_us,p95_us,plan_build_us,plan_validate_us,acquire_us,release_us,matmul_us,add_us,softmax_us,transpose_us,allocations_per_execution,reuses_per_execution,releases_per_execution,peak_live_slots,peak_live_bytes");
    println!(
        "numeric,current_allocating_chain,{:.3},{:.3},0.000,0.000,0.000,0.000,{:.3},{:.3},{:.3},{:.3},4,0,3,2,{}",
        baseline.median.as_secs_f64() * 1e6,
        baseline.p95.as_secs_f64() * 1e6,
        baseline_stages[0].as_secs_f64() * 1e6,
        baseline_stages[1].as_secs_f64() * 1e6,
        baseline_stages[2].as_secs_f64() * 1e6,
        baseline_stages[3].as_secs_f64() * 1e6,
        2 * m * n * size_of::<f32>(),
    );
    println!(
        "numeric,graph_executor,{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{},{},{},{},{}",
        planned.median.as_secs_f64() * 1e6,
        planned.p95.as_secs_f64() * 1e6,
        plan_build.median.as_secs_f64() * 1e6,
        plan_validate.median.as_secs_f64() * 1e6,
        result.lifecycle_timings.acquire.as_secs_f64() * 1e6,
        result.lifecycle_timings.release.as_secs_f64() * 1e6,
        result.timings.matmul.as_secs_f64() * 1e6,
        result.timings.add.as_secs_f64() * 1e6,
        result.timings.softmax.as_secs_f64() * 1e6,
        result.timings.transpose.as_secs_f64() * 1e6,
        result.metrics.physical_allocations,
        result.metrics.reuses,
        result.metrics.releases,
        result.metrics.peak_live_slots,
        result.metrics.peak_live_bytes,
    );
}

fn run_graph_pipeline(
    graph: &Graph,
    storage: &graph::StoragePlan,
    plan: &ExecutionPlan,
    ids: [ValueId; 3],
    a: &[f32],
    b: &[f32],
    residual: &[f32],
) -> graph::ExecutionResult {
    let mut external = ExternalInputs::new(graph.values.len());
    for (id, shape, data) in [
        (ids[0], graph.values[ids[0].0].shape.as_slice(), a),
        (ids[1], graph.values[ids[1].0].shape.as_slice(), b),
        (ids[2], graph.values[ids[2].0].shape.as_slice(), residual),
    ] {
        external
            .bind(
                id,
                ExternalTensor {
                    dtype: GraphDType::Float32,
                    layout: Layout::ContiguousRowMajor,
                    shape,
                    data,
                },
            )
            .unwrap();
    }
    execute(graph, storage, plan, &external).unwrap()
}

fn run_allocating_pipeline(
    a: &[f32],
    b: &[f32],
    residual: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    run_allocating_pipeline_timed(a, b, residual, m, k, n).0
}

fn run_allocating_pipeline_timed(
    a: &[f32],
    b: &[f32],
    residual: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> (Vec<f32>, [Duration; 4]) {
    let mut multiplied = vec![0.0; m * n];
    let started = Instant::now();
    matmul_dispatch::matmul_dispatch_f32(a, b, &mut multiplied, m, k, n).unwrap();
    let matmul = started.elapsed();
    let mut added = vec![0.0; m * n];
    let started = Instant::now();
    add::add_f32(&multiplied, residual, &mut added).unwrap();
    let add = started.elapsed();
    let mut normalized = vec![0.0; m * n];
    let started = Instant::now();
    softmax::softmax_last_dim_f32(&added, &mut normalized, n).unwrap();
    let softmax = started.elapsed();
    let mut output = vec![0.0; m * n];
    let started = Instant::now();
    transpose::transpose_f32(&normalized, &mut output, m, n).unwrap();
    let transpose = started.elapsed();
    (output, [matmul, add, softmax, transpose])
}

fn deterministic_values(length: usize, scale: f32) -> Vec<f32> {
    (0..length)
        .map(|index| ((index % 251) as f32 - 125.0) * scale)
        .collect()
}

#[derive(Clone, Copy)]
enum GraphCase {
    Transformer,
    Linear(usize),
    Branching,
}

impl GraphCase {
    fn build(self) -> Graph {
        match self {
            Self::Transformer => build_transformer_unanalyzed(128, 768, 768).0,
            Self::Linear(nodes) => build_linear(nodes),
            Self::Branching => build_branching(),
        }
    }
}

fn build_transformer_shape(m: usize, k: usize, n: usize) -> (Graph, [ValueId; 3]) {
    let (mut graph, ids) = build_transformer_unanalyzed(m, k, n);
    graph.analyze_lifetimes();
    (graph, ids)
}

fn build_transformer_unanalyzed(m: usize, k: usize, n: usize) -> (Graph, [ValueId; 3]) {
    let mut builder = GraphBuilder::new();
    let activations = input(&mut builder, &[m, k]);
    let weights = input(&mut builder, &[k, n]);
    let residual = input(&mut builder, &[m, n]);
    let multiplied = builder
        .operation(Operation::Matmul, &[activations, weights])
        .unwrap();
    let added = builder
        .operation(Operation::Add, &[multiplied, residual])
        .unwrap();
    let normalized = builder
        .operation(Operation::SoftmaxLastDim, &[added])
        .unwrap();
    let output = builder
        .operation(Operation::Transpose, &[normalized])
        .unwrap();
    builder.mark_external_output(output).unwrap();
    (builder.build(), [activations, weights, residual])
}

fn benchmark_case(name: &str, case: GraphCase, samples: usize, target: Duration) {
    let construction = measure(samples, target, || black_box(case.build()));

    let mut lifetime_graph = case.build();
    let lifetime = measure(samples, target, || {
        lifetime_graph.analyze_lifetimes();
        black_box(&lifetime_graph);
    });

    let mut graph = case.build();
    graph.analyze_lifetimes();
    let planning = measure(samples, target, || {
        black_box(plan_storage(black_box(&graph)).unwrap());
    });
    let plan = plan_storage(&graph).unwrap();
    let execution_plan = ExecutionPlan::new(&graph, &plan).unwrap();
    let validation = measure(samples, target, || {
        execution_plan
            .validate(black_box(&graph), black_box(&plan))
            .unwrap();
        black_box(());
    });
    let execution = measure(samples, target, || {
        black_box(simulate(black_box(&graph), black_box(&plan)).unwrap());
    });
    let simulation = simulate(&graph, &plan).unwrap();

    for (stage, measurement) in [
        ("construction", construction),
        ("lifetime", lifetime),
        ("storage_plan", planning),
        ("plan_validate", validation),
        ("simulation", execution),
    ] {
        println!(
            "{name},{stage},{},{:.3},{:.3},{},{},{},{},{}",
            graph.nodes.len(),
            measurement.median.as_secs_f64() * 1e6,
            measurement.p95.as_secs_f64() * 1e6,
            measurement.iterations,
            plan.slots.len(),
            simulation.metrics.physical_allocations,
            simulation.metrics.reuses,
            simulation.metrics.peak_live_bytes,
        );
    }
}

fn build_linear(nodes: usize) -> Graph {
    assert!(nodes > 0);
    let mut builder = GraphBuilder::new();
    let mut current = input(&mut builder, &[128, 768]);
    for index in 0..nodes {
        let operation = if index % 2 == 0 {
            Operation::SoftmaxLastDim
        } else {
            Operation::Transpose
        };
        current = builder.operation(operation, &[current]).unwrap();
    }
    builder.mark_external_output(current).unwrap();
    builder.build()
}

fn build_branching() -> Graph {
    let mut builder = GraphBuilder::new();
    let left = input(&mut builder, &[128, 768]);
    let right = input(&mut builder, &[768, 768]);
    let x = builder
        .operation(Operation::Matmul, &[left, right])
        .unwrap();
    let first = builder.operation(Operation::SoftmaxLastDim, &[x]).unwrap();
    let second = builder.operation(Operation::Transpose, &[x]).unwrap();
    let first_transposed = builder.operation(Operation::Transpose, &[first]).unwrap();
    let output = builder
        .operation(Operation::Add, &[first_transposed, second])
        .unwrap();
    builder.mark_external_output(output).unwrap();
    builder.build()
}

fn input(builder: &mut GraphBuilder, shape: &[usize]) -> ValueId {
    builder
        .input(
            GraphDType::Float32,
            shape.to_vec(),
            Layout::ContiguousRowMajor,
        )
        .unwrap()
}

fn measure<F, T>(samples: usize, target: Duration, mut operation: F) -> Measurement
where
    F: FnMut() -> T,
{
    let calibration_start = Instant::now();
    black_box(operation());
    let calibration = calibration_start.elapsed().max(Duration::from_nanos(1));
    let iterations = ((target.as_nanos() / calibration.as_nanos()).max(1) as usize).min(100_000);
    let mut durations = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(operation());
        }
        durations.push(start.elapsed().div_f64(iterations as f64));
    }
    durations.sort_unstable();
    Measurement {
        median: durations[(durations.len() - 1) / 2],
        p95: durations[((durations.len() - 1) as f64 * 0.95).ceil() as usize],
        iterations,
    }
}

fn measure_pair<F, G, T, U>(
    samples: usize,
    target: Duration,
    mut first: F,
    mut second: G,
) -> (Measurement, Measurement)
where
    F: FnMut() -> T,
    G: FnMut() -> U,
{
    let started = Instant::now();
    black_box(first());
    let first_calibration = started.elapsed().max(Duration::from_nanos(1));
    let started = Instant::now();
    black_box(second());
    let second_calibration = started.elapsed().max(Duration::from_nanos(1));
    let slowest = first_calibration.max(second_calibration);
    let iterations = ((target.as_nanos() / slowest.as_nanos()).max(1) as usize).min(100_000);
    let mut first_samples = Vec::with_capacity(samples);
    let mut second_samples = Vec::with_capacity(samples);
    for sample in 0..samples {
        let run_first = |operation: &mut dyn FnMut()| {
            let started = Instant::now();
            for _ in 0..iterations {
                operation();
            }
            started.elapsed().div_f64(iterations as f64)
        };
        let mut first_operation = || {
            black_box(first());
        };
        let mut second_operation = || {
            black_box(second());
        };
        if sample % 2 == 0 {
            first_samples.push(run_first(&mut first_operation));
            second_samples.push(run_first(&mut second_operation));
        } else {
            second_samples.push(run_first(&mut second_operation));
            first_samples.push(run_first(&mut first_operation));
        }
    }
    (
        summarize(first_samples, iterations),
        summarize(second_samples, iterations),
    )
}

fn summarize(mut samples: Vec<Duration>, iterations: usize) -> Measurement {
    samples.sort_unstable();
    Measurement {
        median: samples[(samples.len() - 1) / 2],
        p95: samples[((samples.len() - 1) as f64 * 0.95).ceil() as usize],
        iterations,
    }
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
