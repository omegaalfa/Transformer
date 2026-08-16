use std::env;
use std::hint::black_box;
use std::time::{Duration, Instant};

#[allow(dead_code)]
#[path = "../src/graph.rs"]
mod graph;

use graph::{plan_storage, simulate, Graph, GraphBuilder, GraphDType, Layout, Operation, ValueId};

const DEFAULT_SAMPLES: usize = 25;
const DEFAULT_TARGET_US: u64 = 1_000;

struct Measurement {
    median: Duration,
    p95: Duration,
    iterations: usize,
}

fn main() {
    let samples = environment_usize("TRANSFORMER_GRAPH_BENCH_SAMPLES", DEFAULT_SAMPLES).max(3);
    let target = Duration::from_micros(environment_u64(
        "TRANSFORMER_GRAPH_BENCH_TARGET_US",
        DEFAULT_TARGET_US,
    ));
    println!(
        "Graph planner benchmarks samples={samples} target_us={}",
        target.as_micros()
    );
    println!("case,stage,nodes,median_us,p95_us,iterations,slots,allocations,reuses,peak_bytes");

    benchmark_case("transformer_4", GraphCase::Transformer, samples, target);
    benchmark_case("linear_10", GraphCase::Linear(10), samples, target);
    benchmark_case("branching", GraphCase::Branching, samples, target);
    benchmark_case("linear_100", GraphCase::Linear(100), samples, target);
    benchmark_case("linear_1000", GraphCase::Linear(1000), samples, target);
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
            Self::Transformer => build_transformer(),
            Self::Linear(nodes) => build_linear(nodes),
            Self::Branching => build_branching(),
        }
    }
}

fn build_transformer() -> Graph {
    let mut builder = GraphBuilder::new();
    let activations = input(&mut builder, &[128, 768]);
    let weights = input(&mut builder, &[768, 768]);
    let residual = input(&mut builder, &[128, 768]);
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
    builder.build()
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
    let execution = measure(samples, target, || {
        black_box(simulate(black_box(&graph), black_box(&plan)).unwrap());
    });
    let simulation = simulate(&graph, &plan).unwrap();

    for (stage, measurement) in [
        ("construction", construction),
        ("lifetime", lifetime),
        ("storage_plan", planning),
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
