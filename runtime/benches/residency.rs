//! Benchmark-only validation of resident immutable Tensor inputs.
//!
//! This intentionally uses the existing Tensor and kernels. It is not linked
//! into the cdylib and introduces no public or FFI contract.

use std::hint::black_box;
use std::time::{Duration, Instant};

#[allow(dead_code, unused_imports)]
#[path = "../src/kernels/add.rs"]
mod add;
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

use tensor::{DType, Shape, Tensor};

const M: usize = 128;
const K: usize = 768;
const N: usize = 768;

#[derive(Clone, Copy, Debug, Default)]
struct AllocationCounts {
    vec_creations: u64,
    tensor_creations: u64,
    handle_creations: u64,
    allocations: u64,
    allocated_bytes: u64,
    copied_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layout {
    ContiguousRowMajor,
    Unsupported,
}

#[derive(Clone, Copy, Debug)]
struct BindingContract<'a> {
    shape: &'a [usize],
    dtype: DType,
    layout: Layout,
    capacity: usize,
    require_non_empty: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BindError {
    Shape,
    DType,
    Layout,
    Capacity,
    Empty,
}

fn validate_binding(tensor: &Tensor, contract: BindingContract<'_>) -> Result<(), BindError> {
    if tensor.dtype() != contract.dtype {
        return Err(BindError::DType);
    }
    if contract.layout != Layout::ContiguousRowMajor
        || tensor.strides().as_slice() != contiguous_strides(tensor.shape().as_slice()).as_slice()
    {
        return Err(BindError::Layout);
    }
    if tensor.shape().as_slice() != contract.shape {
        return Err(BindError::Shape);
    }
    if contract.capacity < tensor.numel() || tensor.as_slice().len() < tensor.numel() {
        return Err(BindError::Capacity);
    }
    if contract.require_non_empty && tensor.is_empty() {
        return Err(BindError::Empty);
    }
    Ok(())
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    if shape.contains(&0) {
        return vec![0; shape.len()];
    }
    let mut result = vec![0; shape.len()];
    let mut stride = 1;
    for axis in (0..shape.len()).rev() {
        result[axis] = stride;
        stride *= shape[axis];
    }
    result
}

struct ResidentBindings<'a> {
    input: &'a Tensor,
    weight: &'a Tensor,
    residual: &'a Tensor,
}

impl<'a> ResidentBindings<'a> {
    fn bind(
        input: &'a Tensor,
        weight: &'a Tensor,
        residual: &'a Tensor,
    ) -> Result<Self, BindError> {
        validate_binding(
            input,
            BindingContract {
                shape: &[M, K],
                dtype: DType::Float32,
                layout: Layout::ContiguousRowMajor,
                capacity: input.as_slice().len(),
                require_non_empty: true,
            },
        )?;
        validate_binding(
            weight,
            BindingContract {
                shape: &[K, N],
                dtype: DType::Float32,
                layout: Layout::ContiguousRowMajor,
                capacity: weight.as_slice().len(),
                require_non_empty: true,
            },
        )?;
        validate_binding(
            residual,
            BindingContract {
                shape: &[M, N],
                dtype: DType::Float32,
                layout: Layout::ContiguousRowMajor,
                capacity: residual.as_slice().len(),
                require_non_empty: true,
            },
        )?;
        Ok(Self {
            input,
            weight,
            residual,
        })
    }

    fn execute(&self, counts: &mut AllocationCounts) -> Tensor {
        let output_shape = Shape::new(vec![M, N]).unwrap();
        let mut multiplied = tracked_zeros(M * N, counts);
        let mut added = tracked_zeros(M * N, counts);
        let mut normalized = tracked_zeros(M * N, counts);
        let mut output = tracked_zeros(M * N, counts);
        matmul_dispatch::matmul_dispatch_f32(
            self.input.as_slice(),
            self.weight.as_slice(),
            &mut multiplied,
            M,
            K,
            N,
        )
        .unwrap();
        add::add_f32(&multiplied, self.residual.as_slice(), &mut added).unwrap();
        softmax::softmax_last_dim_f32(&added, &mut normalized, N).unwrap();
        transpose::transpose_f32(&normalized, &mut output, M, N).unwrap();
        // The existing FFI Tensor pipeline wraps every one of the four output
        // Vecs in a Tensor/handle; three are destroyed before returning O.
        counts.tensor_creations += 4;
        counts.handle_creations += 4;
        Tensor::from_vec(output, output_shape).unwrap()
    }
}

fn tracked_zeros(length: usize, counts: &mut AllocationCounts) -> Vec<f32> {
    counts.vec_creations += 1;
    counts.allocations += 1;
    counts.allocated_bytes += (length * size_of::<f32>()) as u64;
    vec![0.0; length]
}

fn tracked_input(values: &[f32], shape: &[usize], counts: &mut AllocationCounts) -> Tensor {
    counts.vec_creations += 1;
    counts.tensor_creations += 1;
    counts.handle_creations += 1;
    counts.allocations += 1;
    counts.allocated_bytes += std::mem::size_of_val(values) as u64;
    counts.copied_bytes += std::mem::size_of_val(values) as u64;
    Tensor::from_vec(values.to_vec(), Shape::new(shape.to_vec()).unwrap()).unwrap()
}

fn values(length: usize, scale: f32) -> Vec<f32> {
    (0..length)
        .map(|index| ((index % 251) as f32 - 125.0) * scale)
        .collect()
}

fn percentile(sorted: &[Duration], percentile: usize) -> Duration {
    sorted[((sorted.len() - 1) * percentile).div_ceil(100)]
}

fn report(label: &str, executions: usize, mut timings: Vec<Duration>, counts: AllocationCounts) {
    timings.sort_unstable();
    println!(
        "residency,{label},{executions},{:.3},{:.3},{:.3},{},{},{},{},{},{}",
        timings[timings.len() / 2].as_secs_f64() * 1e6,
        percentile(&timings, 95).as_secs_f64() * 1e6,
        percentile(&timings, 99).as_secs_f64() * 1e6,
        counts.allocations,
        counts.allocated_bytes,
        counts.copied_bytes,
        counts.vec_creations,
        counts.tensor_creations,
        counts.handle_creations,
    );
}

fn validate_contract_and_isolation(a: &[f32], b: &[f32], residual: &[f32]) {
    let mut counts = AllocationCounts::default();
    let input = tracked_input(a, &[M, K], &mut counts);
    let weight = tracked_input(b, &[K, N], &mut counts);
    let residual_tensor = tracked_input(residual, &[M, N], &mut counts);
    let resident = ResidentBindings::bind(&input, &weight, &residual_tensor).unwrap();
    let first = resident.execute(&mut counts);
    let second = resident.execute(&mut counts);
    assert_eq!(first.as_slice(), second.as_slice());

    let mut other_counts = AllocationCounts::default();
    let other_input = tracked_input(a, &[M, K], &mut other_counts);
    let other_weight = tracked_input(b, &[K, N], &mut other_counts);
    let other_residual = tracked_input(residual, &[M, N], &mut other_counts);
    let other = ResidentBindings::bind(&other_input, &other_weight, &other_residual).unwrap();
    assert_eq!(
        first.as_slice(),
        other.execute(&mut other_counts).as_slice()
    );

    assert_eq!(
        validate_binding(
            &input,
            BindingContract {
                shape: &[M, K + 1],
                dtype: DType::Float32,
                layout: Layout::ContiguousRowMajor,
                capacity: input.numel(),
                require_non_empty: true,
            }
        ),
        Err(BindError::Shape)
    );
    assert_eq!(
        validate_binding(
            &input,
            BindingContract {
                shape: &[M, K],
                dtype: DType::Float32,
                layout: Layout::Unsupported,
                capacity: input.numel(),
                require_non_empty: true,
            }
        ),
        Err(BindError::Layout)
    );
    assert_eq!(
        validate_binding(
            &input,
            BindingContract {
                shape: &[M, K],
                dtype: DType::Float32,
                layout: Layout::ContiguousRowMajor,
                capacity: input.numel() - 1,
                require_non_empty: true,
            }
        ),
        Err(BindError::Capacity)
    );
}

fn benchmark_resident_phases(samples: usize, a: &[f32], b: &[f32], residual: &[f32]) {
    let mut setup_timings = Vec::with_capacity(samples);
    let mut first_timings = Vec::with_capacity(samples);
    let mut second_timings = Vec::with_capacity(samples);
    let mut teardown_timings = Vec::with_capacity(samples);
    let mut setup_counts = AllocationCounts::default();
    let mut execute_counts = AllocationCounts::default();

    for _ in 0..samples {
        let mut counts = AllocationCounts::default();
        let started = Instant::now();
        let input = tracked_input(a, &[M, K], &mut counts);
        let weight = tracked_input(b, &[K, N], &mut counts);
        let residual_tensor = tracked_input(residual, &[M, N], &mut counts);
        let binding = ResidentBindings::bind(&input, &weight, &residual_tensor).unwrap();
        setup_timings.push(started.elapsed());
        setup_counts = counts;

        let mut first_counts = AllocationCounts::default();
        let started = Instant::now();
        let first = binding.execute(&mut first_counts);
        first_timings.push(started.elapsed());
        execute_counts = first_counts;

        let mut second_counts = AllocationCounts::default();
        let started = Instant::now();
        let second = binding.execute(&mut second_counts);
        second_timings.push(started.elapsed());
        assert_eq!(first.as_slice(), second.as_slice());

        let started = Instant::now();
        drop(first);
        drop(second);
        drop(residual_tensor);
        drop(weight);
        drop(input);
        teardown_timings.push(started.elapsed());
    }

    report("B_setup", 1, setup_timings, setup_counts);
    report("B_first_execute", 1, first_timings, execute_counts);
    report("B_second_execute", 1, second_timings, execute_counts);
    report(
        "B_teardown",
        1,
        teardown_timings,
        AllocationCounts::default(),
    );
}

fn main() {
    let samples = std::env::var("TRANSFORMER_RESIDENCY_SAMPLES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(25_usize)
        .max(25);
    let maximum_executions = std::env::var("TRANSFORMER_RESIDENCY_MAX_N")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1000_usize);
    let a = values(M * K, 0.001);
    let b = values(K * N, 0.002);
    let residual = values(M * N, 0.003);
    validate_contract_and_isolation(&a, &b, &residual);
    println!("residency,model,executions,p50_us,p95_us,p99_us,allocations,allocated_bytes,copied_bytes,vec_creations,tensor_creations,handle_creations");
    benchmark_resident_phases(samples, &a, &b, &residual);

    for executions in [1, 10, 100, 1000]
        .into_iter()
        .filter(|executions| *executions <= maximum_executions)
    {
        let mut recreate_timings = Vec::with_capacity(samples);
        let mut resident_timings = Vec::with_capacity(samples);
        let mut recreate_counts = AllocationCounts::default();
        let mut resident_counts = AllocationCounts::default();
        for sample in 0..samples {
            let run_recreate = || {
                let mut counts = AllocationCounts::default();
                let started = Instant::now();
                for _ in 0..executions {
                    let input = tracked_input(&a, &[M, K], &mut counts);
                    let weight = tracked_input(&b, &[K, N], &mut counts);
                    let residual_tensor = tracked_input(&residual, &[M, N], &mut counts);
                    let binding =
                        ResidentBindings::bind(&input, &weight, &residual_tensor).unwrap();
                    black_box(binding.execute(&mut counts));
                }
                (started.elapsed(), counts)
            };
            let run_resident = || {
                let mut counts = AllocationCounts::default();
                let started = Instant::now();
                let input = tracked_input(&a, &[M, K], &mut counts);
                let weight = tracked_input(&b, &[K, N], &mut counts);
                let residual_tensor = tracked_input(&residual, &[M, N], &mut counts);
                let binding = ResidentBindings::bind(&input, &weight, &residual_tensor).unwrap();
                for _ in 0..executions {
                    black_box(binding.execute(&mut counts));
                }
                (started.elapsed(), counts)
            };
            let ((recreate, recreate_sample_counts), (resident, resident_sample_counts)) =
                if sample % 2 == 0 {
                    (run_recreate(), run_resident())
                } else {
                    let resident = run_resident();
                    let recreate = run_recreate();
                    (recreate, resident)
                };
            recreate_timings.push(recreate);
            resident_timings.push(resident);
            recreate_counts = recreate_sample_counts;
            resident_counts = resident_sample_counts;
        }
        report("A_recreate", executions, recreate_timings, recreate_counts);
        report("B_resident", executions, resident_timings, resident_counts);
    }
}
