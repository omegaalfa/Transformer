use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::{Duration, Instant};

use crate::kernels::add::add_f32;
use crate::kernels::matmul_dispatch::matmul_dispatch_f32;
use crate::kernels::softmax::softmax_last_dim_f32;
use crate::kernels::transpose::transpose_f32;

use super::{
    Graph, GraphDType, Layout, NodeId, Operation, PoolMetrics, StoragePlan, StorageSlotId, Value,
    ValueId,
};

#[cfg(test)]
#[derive(Clone, Copy)]
enum FailureInjection {
    Error,
    Panic,
}

#[cfg(test)]
thread_local! {
    static FAILURE_INJECTION: std::cell::Cell<Option<FailureInjection>> = const { std::cell::Cell::new(None) };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InputBinding {
    ExternalInput(ValueId),
    Slot(ValueId, StorageSlotId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutputBinding {
    Slot(ValueId, StorageSlotId),
    ExternalOutput(ValueId, StorageSlotId),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NodeExecution {
    pub(crate) node: NodeId,
    pub(crate) operation: Operation,
    pub(crate) inputs: Vec<InputBinding>,
    pub(crate) output: OutputBinding,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExecutionPlan {
    pub(crate) nodes: Vec<NodeExecution>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExecutionError {
    GraphNotAnalyzed,
    InvalidPlan,
    MissingExternalInput,
    IncompatibleDType,
    IncompatibleLayout,
    IncompatibleShape,
    InsufficientCapacity,
    SlotUnavailable,
    ValueUnavailable,
    KernelFailure,
    KernelPanic,
}

impl ExecutionPlan {
    pub(crate) fn new(graph: &Graph, storage: &StoragePlan) -> Result<Self, ExecutionError> {
        if !graph.is_analyzed() || storage.assignments.len() != graph.values.len() {
            return Err(ExecutionError::GraphNotAnalyzed);
        }
        let mut nodes = Vec::with_capacity(graph.nodes.len());
        for (index, node) in graph.nodes.iter().enumerate() {
            if node.id != NodeId(index)
                || graph
                    .values
                    .get(node.output.0)
                    .and_then(|value| value.producer)
                    != Some(node.id)
            {
                return Err(ExecutionError::InvalidPlan);
            }
            let inputs = node
                .inputs
                .iter()
                .map(|input| {
                    let value = graph
                        .values
                        .get(input.0)
                        .ok_or(ExecutionError::InvalidPlan)?;
                    if value.external_input {
                        Ok(InputBinding::ExternalInput(*input))
                    } else {
                        let slot = assignment(storage, *input)?;
                        Ok(InputBinding::Slot(*input, slot))
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            let output_value = &graph.values[node.output.0];
            let output_slot = assignment(storage, node.output)?;
            validate_slot(output_value, output_slot, storage)?;
            let output = if output_value.external_output {
                OutputBinding::ExternalOutput(node.output, output_slot)
            } else {
                OutputBinding::Slot(node.output, output_slot)
            };
            nodes.push(NodeExecution {
                node: node.id,
                operation: node.operation,
                inputs,
                output,
            });
        }
        let plan = Self { nodes };
        plan.validate(graph, storage)?;
        Ok(plan)
    }

    pub(crate) fn validate(
        &self,
        graph: &Graph,
        storage: &StoragePlan,
    ) -> Result<(), ExecutionError> {
        if self.nodes.len() != graph.nodes.len() {
            return Err(ExecutionError::InvalidPlan);
        }
        for (index, slot) in storage.slots.iter().enumerate() {
            if slot.id != StorageSlotId(index) {
                return Err(ExecutionError::InvalidPlan);
            }
            for (position, left_id) in slot.values.iter().enumerate() {
                let left = graph
                    .values
                    .get(left_id.0)
                    .ok_or(ExecutionError::InvalidPlan)?;
                if assignment(storage, *left_id)? != slot.id {
                    return Err(ExecutionError::InvalidPlan);
                }
                validate_slot(left, slot.id, storage)?;
                for right_id in &slot.values[position + 1..] {
                    let right = graph
                        .values
                        .get(right_id.0)
                        .ok_or(ExecutionError::InvalidPlan)?;
                    if left
                        .lifetime
                        .ok_or(ExecutionError::InvalidPlan)?
                        .overlaps(right.lifetime.ok_or(ExecutionError::InvalidPlan)?)
                    {
                        return Err(ExecutionError::InvalidPlan);
                    }
                }
            }
        }
        for (index, execution) in self.nodes.iter().enumerate() {
            let node = graph.nodes.get(index).ok_or(ExecutionError::InvalidPlan)?;
            if execution.node != NodeId(index)
                || execution.node != node.id
                || execution.operation != node.operation
                || execution.inputs.len() != node.inputs.len()
                || output_value(execution.output) != node.output
            {
                return Err(ExecutionError::InvalidPlan);
            }
            for (binding, expected) in execution.inputs.iter().zip(&node.inputs) {
                if input_value(*binding) != *expected {
                    return Err(ExecutionError::InvalidPlan);
                }
                let value = graph
                    .values
                    .get(expected.0)
                    .ok_or(ExecutionError::InvalidPlan)?;
                if let Some(producer) = value.producer {
                    if producer >= node.id {
                        return Err(ExecutionError::InvalidPlan);
                    }
                }
                match binding {
                    InputBinding::ExternalInput(_) if value.external_input => {}
                    InputBinding::Slot(_, slot) if !value.external_input => {
                        if assignment(storage, *expected)? != *slot {
                            return Err(ExecutionError::InvalidPlan);
                        }
                        validate_slot(value, *slot, storage)?;
                    }
                    _ => return Err(ExecutionError::InvalidPlan),
                }
            }
            let output = &graph.values[node.output.0];
            let expected_slot = assignment(storage, node.output)?;
            if !storage.slots[expected_slot.0].values.contains(&node.output) {
                return Err(ExecutionError::InvalidPlan);
            }
            match execution.output {
                OutputBinding::Slot(_, slot)
                    if !output.external_output && slot == expected_slot => {}
                OutputBinding::ExternalOutput(_, slot)
                    if output.external_output && slot == expected_slot => {}
                _ => return Err(ExecutionError::InvalidPlan),
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ExternalTensor<'a> {
    pub(crate) dtype: GraphDType,
    pub(crate) layout: Layout,
    pub(crate) shape: &'a [usize],
    pub(crate) data: &'a [f32],
}

#[derive(Debug)]
pub(crate) struct ExternalInputs<'a> {
    values: Vec<Option<ExternalTensor<'a>>>,
}

impl<'a> ExternalInputs<'a> {
    pub(crate) fn new(value_count: usize) -> Self {
        Self {
            values: vec![None; value_count],
        }
    }

    pub(crate) fn bind(
        &mut self,
        value: ValueId,
        tensor: ExternalTensor<'a>,
    ) -> Result<(), ExecutionError> {
        let destination = self
            .values
            .get_mut(value.0)
            .ok_or(ExecutionError::InvalidPlan)?;
        *destination = Some(tensor);
        Ok(())
    }

    fn get(&self, value: ValueId) -> Result<ExternalTensor<'a>, ExecutionError> {
        self.values
            .get(value.0)
            .copied()
            .flatten()
            .ok_or(ExecutionError::MissingExternalInput)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct KernelTimings {
    pub(crate) matmul: Duration,
    pub(crate) add: Duration,
    pub(crate) softmax: Duration,
    pub(crate) transpose: Duration,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct LifecycleTimings {
    pub(crate) acquire: Duration,
    pub(crate) release: Duration,
}

#[derive(Debug, PartialEq)]
pub(crate) struct ExecutedValue {
    pub(crate) id: ValueId,
    pub(crate) dtype: GraphDType,
    pub(crate) layout: Layout,
    pub(crate) shape: Vec<usize>,
    pub(crate) data: Vec<f32>,
}

#[derive(Debug, PartialEq)]
pub(crate) struct ExecutionResult {
    pub(crate) outputs: Vec<ExecutedValue>,
    pub(crate) metrics: PoolMetrics,
    pub(crate) timings: KernelTimings,
    pub(crate) lifecycle_timings: LifecycleTimings,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotState {
    Available,
    Borrowed(NodeId),
    Bound(ValueId),
    PinnedExternal(ValueId),
    Delivered,
}

struct PhysicalSlot {
    id: StorageSlotId,
    dtype: GraphDType,
    layout: Layout,
    capacity_elements: usize,
    buffer: Vec<f32>,
    state: SlotState,
    acquisitions: usize,
}

struct PhysicalStoragePool {
    slots: Vec<PhysicalSlot>,
    metrics: PoolMetrics,
    live_slots: usize,
    live_bytes: usize,
}

struct DestinationLease {
    slot: StorageSlotId,
    node: NodeId,
    dtype: GraphDType,
    layout: Layout,
    capacity_elements: usize,
    buffer: Vec<f32>,
}

impl DestinationLease {
    fn as_exact_f32_slice_mut(&mut self, numel: usize) -> Result<&mut [f32], ExecutionError> {
        if self.dtype != GraphDType::Float32 {
            return Err(ExecutionError::IncompatibleDType);
        }
        if self.layout != Layout::ContiguousRowMajor {
            return Err(ExecutionError::IncompatibleLayout);
        }
        if self.capacity_elements < numel || self.buffer.len() < numel {
            return Err(ExecutionError::InsufficientCapacity);
        }
        Ok(&mut self.buffer[..numel])
    }
}

impl PhysicalStoragePool {
    fn new(plan: &StoragePlan) -> Result<Self, ExecutionError> {
        let mut metrics = PoolMetrics::default();
        let slots = plan
            .slots
            .iter()
            .map(|slot| {
                let element_size = match slot.dtype {
                    GraphDType::Float32 => size_of::<f32>(),
                    GraphDType::Float64 => return Err(ExecutionError::IncompatibleDType),
                };
                if slot.layout != Layout::ContiguousRowMajor
                    || slot.capacity_bytes % element_size != 0
                {
                    return Err(ExecutionError::IncompatibleLayout);
                }
                let capacity_elements = slot.capacity_bytes / element_size;
                metrics.physical_allocations += 1;
                metrics.bytes_allocated += slot.capacity_bytes;
                Ok(PhysicalSlot {
                    id: slot.id,
                    dtype: slot.dtype,
                    layout: slot.layout,
                    capacity_elements,
                    buffer: vec![0.0; capacity_elements],
                    state: SlotState::Available,
                    acquisitions: 0,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            slots,
            metrics,
            live_slots: 0,
            live_bytes: 0,
        })
    }

    fn acquire(
        &mut self,
        slot_id: StorageSlotId,
        node: NodeId,
        value: &Value,
    ) -> Result<DestinationLease, ExecutionError> {
        let slot = self
            .slots
            .get_mut(slot_id.0)
            .ok_or(ExecutionError::InvalidPlan)?;
        if slot.id != slot_id || slot.state != SlotState::Available {
            return Err(ExecutionError::SlotUnavailable);
        }
        if slot.dtype != value.dtype {
            return Err(ExecutionError::IncompatibleDType);
        }
        if slot.layout != value.layout {
            return Err(ExecutionError::IncompatibleLayout);
        }
        if slot.capacity_elements < value.numel {
            return Err(ExecutionError::InsufficientCapacity);
        }
        slot.state = SlotState::Borrowed(node);
        if slot.acquisitions > 0 {
            self.metrics.reuses += 1;
            self.metrics.bytes_reused += value.bytes;
        }
        slot.acquisitions += 1;
        self.live_slots += 1;
        self.live_bytes += slot.capacity_elements * size_of::<f32>();
        self.metrics.peak_live_slots = self.metrics.peak_live_slots.max(self.live_slots);
        self.metrics.peak_live_bytes = self.metrics.peak_live_bytes.max(self.live_bytes);
        Ok(DestinationLease {
            slot: slot_id,
            node,
            dtype: slot.dtype,
            layout: slot.layout,
            capacity_elements: slot.capacity_elements,
            buffer: std::mem::take(&mut slot.buffer),
        })
    }

    fn commit(&mut self, lease: DestinationLease, value: &Value) -> Result<(), ExecutionError> {
        let slot = self
            .slots
            .get_mut(lease.slot.0)
            .ok_or(ExecutionError::InvalidPlan)?;
        if slot.state != SlotState::Borrowed(lease.node) {
            return Err(ExecutionError::SlotUnavailable);
        }
        slot.buffer = lease.buffer;
        slot.state = SlotState::Bound(value.id);
        Ok(())
    }

    fn pin(&mut self, value: ValueId, slot: StorageSlotId) -> Result<(), ExecutionError> {
        let slot = self
            .slots
            .get_mut(slot.0)
            .ok_or(ExecutionError::InvalidPlan)?;
        if slot.state != SlotState::Bound(value) {
            return Err(ExecutionError::ValueUnavailable);
        }
        slot.state = SlotState::PinnedExternal(value);
        Ok(())
    }

    fn abort(&mut self, lease: DestinationLease) -> Result<(), ExecutionError> {
        let slot = self
            .slots
            .get_mut(lease.slot.0)
            .ok_or(ExecutionError::InvalidPlan)?;
        if slot.state != SlotState::Borrowed(lease.node) {
            return Err(ExecutionError::SlotUnavailable);
        }
        slot.buffer = lease.buffer;
        slot.state = SlotState::Available;
        self.live_slots -= 1;
        self.live_bytes -= slot.capacity_elements * size_of::<f32>();
        Ok(())
    }

    fn value_slice(
        &self,
        value: ValueId,
        slot: StorageSlotId,
        numel: usize,
    ) -> Result<&[f32], ExecutionError> {
        let slot = self.slots.get(slot.0).ok_or(ExecutionError::InvalidPlan)?;
        if !matches!(
            slot.state,
            SlotState::Bound(bound) | SlotState::PinnedExternal(bound) if bound == value
        ) {
            return Err(ExecutionError::ValueUnavailable);
        }
        slot.buffer
            .get(..numel)
            .ok_or(ExecutionError::InsufficientCapacity)
    }

    fn release(&mut self, value: ValueId, slot: StorageSlotId) -> Result<(), ExecutionError> {
        let slot = self
            .slots
            .get_mut(slot.0)
            .ok_or(ExecutionError::InvalidPlan)?;
        if slot.state != SlotState::Bound(value) {
            return Err(ExecutionError::ValueUnavailable);
        }
        slot.state = SlotState::Available;
        self.metrics.releases += 1;
        self.live_slots -= 1;
        self.live_bytes -= slot.capacity_elements * size_of::<f32>();
        Ok(())
    }

    fn deliver(&mut self, value: &Value, slot: StorageSlotId) -> Result<Vec<f32>, ExecutionError> {
        let slot = self
            .slots
            .get_mut(slot.0)
            .ok_or(ExecutionError::InvalidPlan)?;
        if slot.state != SlotState::PinnedExternal(value.id) {
            return Err(ExecutionError::ValueUnavailable);
        }
        slot.state = SlotState::Delivered;
        self.live_slots -= 1;
        self.live_bytes -= slot.capacity_elements * size_of::<f32>();
        let mut data = std::mem::take(&mut slot.buffer);
        data.truncate(value.numel);
        Ok(data)
    }
}

pub(crate) fn execute(
    graph: &Graph,
    storage: &StoragePlan,
    plan: &ExecutionPlan,
    external: &ExternalInputs<'_>,
) -> Result<ExecutionResult, ExecutionError> {
    plan.validate(graph, storage)?;
    validate_external_inputs(graph, external)?;
    let mut pool = PhysicalStoragePool::new(storage)?;
    let mut timings = KernelTimings::default();
    let mut lifecycle_timings = LifecycleTimings::default();

    for execution in &plan.nodes {
        let node = &graph.nodes[execution.node.0];
        let output = &graph.values[node.output.0];
        let output_slot = output_slot(execution.output);
        let acquire_started = Instant::now();
        let mut lease = pool.acquire(output_slot, node.id, output)?;
        lifecycle_timings.acquire += acquire_started.elapsed();
        let started = Instant::now();
        let kernel_result = catch_unwind(AssertUnwindSafe(|| {
            run_kernel(
                graph,
                node.operation,
                &execution.inputs,
                external,
                &pool,
                &mut lease,
                output.numel,
            )
        }));
        let elapsed = started.elapsed();
        match node.operation {
            Operation::Matmul => timings.matmul += elapsed,
            Operation::Add => timings.add += elapsed,
            Operation::SoftmaxLastDim => timings.softmax += elapsed,
            Operation::Transpose => timings.transpose += elapsed,
        }
        match kernel_result {
            Ok(Ok(())) => {}
            Ok(Err(_)) => {
                pool.abort(lease)?;
                return Err(ExecutionError::KernelFailure);
            }
            Err(_) => {
                pool.abort(lease)?;
                return Err(ExecutionError::KernelPanic);
            }
        }
        pool.commit(lease, output)?;
        if output.external_output {
            pool.pin(output.id, output_slot)?;
        }

        for binding in &execution.inputs {
            let value_id = input_value(*binding);
            let value = &graph.values[value_id.0];
            if value.external_input || value.external_output || value.last_use != Some(node.id) {
                continue;
            }
            if let InputBinding::Slot(_, slot) = binding {
                let release_started = Instant::now();
                pool.release(value_id, *slot)?;
                lifecycle_timings.release += release_started.elapsed();
            }
        }
    }

    let mut outputs = Vec::new();
    for value in graph.values.iter().filter(|value| value.external_output) {
        let slot = assignment(storage, value.id)?;
        outputs.push(ExecutedValue {
            id: value.id,
            dtype: value.dtype,
            layout: value.layout,
            shape: value.shape.clone(),
            data: pool.deliver(value, slot)?,
        });
    }
    Ok(ExecutionResult {
        outputs,
        metrics: pool.metrics,
        timings,
        lifecycle_timings,
    })
}

fn run_kernel(
    graph: &Graph,
    operation: Operation,
    bindings: &[InputBinding],
    external: &ExternalInputs<'_>,
    pool: &PhysicalStoragePool,
    output: &mut DestinationLease,
    output_numel: usize,
) -> Result<(), ExecutionError> {
    #[cfg(test)]
    FAILURE_INJECTION.with(|injection| match injection.take() {
        Some(FailureInjection::Error) => Err(ExecutionError::KernelFailure),
        Some(FailureInjection::Panic) => panic!("controlled graph kernel panic"),
        None => Ok(()),
    })?;
    match (operation, bindings) {
        (Operation::Matmul, [left, right]) => {
            let left_value = &graph.values[input_value(*left).0];
            let right_value = &graph.values[input_value(*right).0];
            let left_data = resolve_input(left_value, *left, external, pool)?;
            let right_data = resolve_input(right_value, *right, external, pool)?;
            let destination = output.as_exact_f32_slice_mut(output_numel)?;
            matmul_dispatch_f32(
                left_data,
                right_data,
                destination,
                left_value.shape[0],
                left_value.shape[1],
                right_value.shape[1],
            )
            .map_err(|_| ExecutionError::KernelFailure)
        }
        (Operation::Add, [left, right]) => {
            let left_value = &graph.values[input_value(*left).0];
            let right_value = &graph.values[input_value(*right).0];
            let left_data = resolve_input(left_value, *left, external, pool)?;
            let right_data = resolve_input(right_value, *right, external, pool)?;
            add_f32(
                left_data,
                right_data,
                output.as_exact_f32_slice_mut(output_numel)?,
            )
            .map_err(|_| ExecutionError::KernelFailure)
        }
        (Operation::SoftmaxLastDim, [input]) => {
            let value = &graph.values[input_value(*input).0];
            let data = resolve_input(value, *input, external, pool)?;
            let last_dim = *value
                .shape
                .last()
                .ok_or(ExecutionError::IncompatibleShape)?;
            softmax_last_dim_f32(data, output.as_exact_f32_slice_mut(output_numel)?, last_dim)
                .map_err(|_| ExecutionError::KernelFailure)
        }
        (Operation::Transpose, [input]) => {
            let value = &graph.values[input_value(*input).0];
            let data = resolve_input(value, *input, external, pool)?;
            transpose_f32(
                data,
                output.as_exact_f32_slice_mut(output_numel)?,
                value.shape[0],
                value.shape[1],
            )
            .map_err(|_| ExecutionError::KernelFailure)
        }
        _ => Err(ExecutionError::InvalidPlan),
    }
}

fn resolve_input<'a>(
    value: &Value,
    binding: InputBinding,
    external: &'a ExternalInputs<'a>,
    pool: &'a PhysicalStoragePool,
) -> Result<&'a [f32], ExecutionError> {
    match binding {
        InputBinding::ExternalInput(id) => Ok(external.get(id)?.data),
        InputBinding::Slot(id, slot) => pool.value_slice(id, slot, value.numel),
    }
}

fn validate_external_inputs(
    graph: &Graph,
    external: &ExternalInputs<'_>,
) -> Result<(), ExecutionError> {
    for value in graph.values.iter().filter(|value| value.external_input) {
        let tensor = external.get(value.id)?;
        if tensor.dtype != value.dtype {
            return Err(ExecutionError::IncompatibleDType);
        }
        if tensor.layout != value.layout {
            return Err(ExecutionError::IncompatibleLayout);
        }
        if tensor.shape != value.shape || tensor.data.len() != value.numel {
            return Err(ExecutionError::IncompatibleShape);
        }
    }
    Ok(())
}

fn validate_slot(
    value: &Value,
    slot: StorageSlotId,
    storage: &StoragePlan,
) -> Result<(), ExecutionError> {
    let slot = storage
        .slots
        .get(slot.0)
        .ok_or(ExecutionError::InvalidPlan)?;
    if slot.dtype != value.dtype {
        return Err(ExecutionError::IncompatibleDType);
    }
    if slot.layout != value.layout {
        return Err(ExecutionError::IncompatibleLayout);
    }
    if slot.capacity_bytes < value.bytes {
        return Err(ExecutionError::InsufficientCapacity);
    }
    Ok(())
}

fn assignment(storage: &StoragePlan, value: ValueId) -> Result<StorageSlotId, ExecutionError> {
    storage
        .assignments
        .get(value.0)
        .copied()
        .flatten()
        .ok_or(ExecutionError::InvalidPlan)
}

fn input_value(binding: InputBinding) -> ValueId {
    match binding {
        InputBinding::ExternalInput(value) | InputBinding::Slot(value, _) => value,
    }
}

fn output_value(binding: OutputBinding) -> ValueId {
    match binding {
        OutputBinding::Slot(value, _) | OutputBinding::ExternalOutput(value, _) => value,
    }
}

fn output_slot(binding: OutputBinding) -> StorageSlotId {
    match binding {
        OutputBinding::Slot(_, slot) | OutputBinding::ExternalOutput(_, slot) => slot,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{plan_storage, GraphBuilder};

    fn input(builder: &mut GraphBuilder, shape: &[usize]) -> ValueId {
        builder
            .input(
                GraphDType::Float32,
                shape.to_vec(),
                Layout::ContiguousRowMajor,
            )
            .unwrap()
    }

    fn transformer_graph(m: usize, k: usize, n: usize) -> (Graph, [ValueId; 3]) {
        let mut builder = GraphBuilder::new();
        let a = input(&mut builder, &[m, k]);
        let b = input(&mut builder, &[k, n]);
        let residual = input(&mut builder, &[m, n]);
        let x = builder.operation(Operation::Matmul, &[a, b]).unwrap();
        let y = builder.operation(Operation::Add, &[x, residual]).unwrap();
        let z = builder.operation(Operation::SoftmaxLastDim, &[y]).unwrap();
        let output = builder.operation(Operation::Transpose, &[z]).unwrap();
        builder.mark_external_output(output).unwrap();
        let mut graph = builder.build();
        graph.analyze_lifetimes();
        (graph, [a, b, residual])
    }

    fn execute_transformer(
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
        residual: &[f32],
    ) -> ExecutionResult {
        let (graph, [a_id, b_id, residual_id]) = transformer_graph(m, k, n);
        let storage = plan_storage(&graph).unwrap();
        let plan = ExecutionPlan::new(&graph, &storage).unwrap();
        let mut external = ExternalInputs::new(graph.values.len());
        let a_shape = [m, k];
        let b_shape = [k, n];
        let residual_shape = [m, n];
        for (id, shape, data) in [
            (a_id, a_shape.as_slice(), a),
            (b_id, b_shape.as_slice(), b),
            (residual_id, residual_shape.as_slice(), residual),
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
        execute(&graph, &storage, &plan, &external).unwrap()
    }

    fn direct_pipeline(
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
        residual: &[f32],
    ) -> Vec<f32> {
        let mut first = vec![0.0; m * n];
        let mut second = vec![0.0; m * n];
        matmul_dispatch_f32(a, b, &mut first, m, k, n).unwrap();
        add_f32(&first, residual, &mut second).unwrap();
        softmax_last_dim_f32(&second, &mut first, n).unwrap();
        transpose_f32(&first, &mut second, m, n).unwrap();
        second
    }

    #[test]
    fn executes_reference_pipeline_with_real_two_slot_reuse() {
        let (m, k, n) = (128, 768, 768);
        let a = vec![0.01; m * k];
        let b = vec![0.02; k * n];
        let residual = vec![0.03; m * n];
        let expected = direct_pipeline(m, k, n, &a, &b, &residual);
        let result = execute_transformer(m, k, n, &a, &b, &residual);
        assert_eq!(result.outputs[0].shape, vec![n, m]);
        assert_eq!(result.outputs[0].dtype, GraphDType::Float32);
        assert_eq!(result.outputs[0].layout, Layout::ContiguousRowMajor);
        assert_eq!(result.outputs[0].data, expected);
        assert_eq!(result.metrics.physical_allocations, 2);
        assert_eq!(result.metrics.reuses, 2);
        assert_eq!(result.metrics.releases, 3);
        assert_eq!(result.metrics.peak_live_slots, 2);
        assert_eq!(result.metrics.peak_live_bytes, 2 * m * n * 4);
    }

    #[test]
    fn matches_direct_kernels_for_all_transformer_shapes() {
        for (k, n) in [(768, 768), (768, 3072), (3072, 768)] {
            for m in [1, 2, 4, 8, 16, 32, 64, 128] {
                let a: Vec<f32> = (0..m * k)
                    .map(|index| (index % 31) as f32 * 0.001 - 0.01)
                    .collect();
                let b: Vec<f32> = (0..k * n)
                    .map(|index| (index % 29) as f32 * 0.001 - 0.02)
                    .collect();
                let residual: Vec<f32> = (0..m * n)
                    .map(|index| (index % 17) as f32 * 0.002 - 0.01)
                    .collect();
                let expected = direct_pipeline(m, k, n, &a, &b, &residual);
                let actual = execute_transformer(m, k, n, &a, &b, &residual);
                assert_eq!(actual.outputs[0].shape, vec![n, m]);
                assert_eq!(actual.outputs[0].data, expected, "M={m}, K={k}, N={n}");
            }
        }
    }

    #[test]
    fn rejects_external_metadata_before_allocating_pool() {
        let (graph, [a_id, b_id, residual_id]) = transformer_graph(2, 2, 2);
        let storage = plan_storage(&graph).unwrap();
        let plan = ExecutionPlan::new(&graph, &storage).unwrap();
        let data = [1.0; 4];
        let mut external = ExternalInputs::new(graph.values.len());
        external
            .bind(
                a_id,
                ExternalTensor {
                    dtype: GraphDType::Float32,
                    layout: Layout::ContiguousColumnMajor,
                    shape: &[2, 2],
                    data: &data,
                },
            )
            .unwrap();
        for id in [b_id, residual_id] {
            external
                .bind(
                    id,
                    ExternalTensor {
                        dtype: GraphDType::Float32,
                        layout: Layout::ContiguousRowMajor,
                        shape: &[2, 2],
                        data: &data,
                    },
                )
                .unwrap();
        }
        assert_eq!(
            execute(&graph, &storage, &plan, &external),
            Err(ExecutionError::IncompatibleLayout)
        );

        let mut wrong_dtype = ExternalInputs::new(graph.values.len());
        for (id, dtype) in [
            (a_id, GraphDType::Float64),
            (b_id, GraphDType::Float32),
            (residual_id, GraphDType::Float32),
        ] {
            wrong_dtype
                .bind(
                    id,
                    ExternalTensor {
                        dtype,
                        layout: Layout::ContiguousRowMajor,
                        shape: &[2, 2],
                        data: &data,
                    },
                )
                .unwrap();
        }
        assert_eq!(
            execute(&graph, &storage, &plan, &wrong_dtype),
            Err(ExecutionError::IncompatibleDType)
        );

        let mut wrong_shape = ExternalInputs::new(graph.values.len());
        for id in [a_id, b_id, residual_id] {
            wrong_shape
                .bind(
                    id,
                    ExternalTensor {
                        dtype: GraphDType::Float32,
                        layout: Layout::ContiguousRowMajor,
                        shape: &[4],
                        data: &data,
                    },
                )
                .unwrap();
        }
        assert_eq!(
            execute(&graph, &storage, &plan, &wrong_shape),
            Err(ExecutionError::IncompatibleShape)
        );
    }

    #[test]
    fn rejects_unanalyzed_graph_and_insufficient_slot_capacity() {
        let mut builder = GraphBuilder::new();
        let input = input(&mut builder, &[2, 2]);
        let output = builder.operation(Operation::Transpose, &[input]).unwrap();
        builder.mark_external_output(output).unwrap();
        let graph = builder.build();
        let empty = StoragePlan {
            slots: Vec::new(),
            assignments: vec![None; graph.values.len()],
        };
        assert_eq!(
            ExecutionPlan::new(&graph, &empty),
            Err(ExecutionError::GraphNotAnalyzed)
        );

        let (graph, _) = transformer_graph(2, 2, 2);
        let mut storage = plan_storage(&graph).unwrap();
        storage.slots[0].capacity_bytes = 4;
        assert_eq!(
            ExecutionPlan::new(&graph, &storage),
            Err(ExecutionError::InsufficientCapacity)
        );

        let (graph, [a_id, b_id, residual_id]) = transformer_graph(2, 2, 2);
        let storage = plan_storage(&graph).unwrap();
        let mut plan = ExecutionPlan::new(&graph, &storage).unwrap();
        plan.nodes[0].node = NodeId(99);
        let data = [1.0; 4];
        let mut external = ExternalInputs::new(graph.values.len());
        for id in [a_id, b_id, residual_id] {
            external
                .bind(
                    id,
                    ExternalTensor {
                        dtype: GraphDType::Float32,
                        layout: Layout::ContiguousRowMajor,
                        shape: &[2, 2],
                        data: &data,
                    },
                )
                .unwrap();
        }
        assert_eq!(
            execute(&graph, &storage, &plan, &external),
            Err(ExecutionError::InvalidPlan)
        );
    }

    #[test]
    fn aborted_lease_is_available_and_does_not_publish_value() {
        let (graph, _) = transformer_graph(2, 2, 2);
        let storage = plan_storage(&graph).unwrap();
        let value = &graph.values[graph.nodes[0].output.0];
        let mut pool = PhysicalStoragePool::new(&storage).unwrap();
        let mut lease = pool.acquire(StorageSlotId(0), NodeId(0), value).unwrap();
        assert_eq!(
            lease.as_exact_f32_slice_mut(value.numel + 1),
            Err(ExecutionError::InsufficientCapacity)
        );
        pool.abort(lease).unwrap();
        assert_eq!(pool.slots[0].state, SlotState::Available);
        assert_eq!(pool.live_slots, 0);
        assert!(matches!(
            pool.value_slice(value.id, StorageSlotId(0), value.numel),
            Err(ExecutionError::ValueUnavailable)
        ));
    }

    #[test]
    fn pool_accepts_only_documented_state_transitions() {
        let (graph, _) = transformer_graph(2, 2, 2);
        let storage = plan_storage(&graph).unwrap();
        let value = &graph.values[graph.nodes[0].output.0];
        let slot = storage.assignments[value.id.0].unwrap();
        let mut pool = PhysicalStoragePool::new(&storage).unwrap();

        assert_eq!(
            pool.release(value.id, slot),
            Err(ExecutionError::ValueUnavailable)
        );
        assert_eq!(
            pool.pin(value.id, slot),
            Err(ExecutionError::ValueUnavailable)
        );
        assert!(matches!(
            pool.deliver(value, slot),
            Err(ExecutionError::ValueUnavailable)
        ));

        let lease = pool.acquire(slot, NodeId(0), value).unwrap();
        assert!(matches!(
            pool.acquire(slot, NodeId(1), value),
            Err(ExecutionError::SlotUnavailable)
        ));
        pool.commit(lease, value).unwrap();
        assert_eq!(pool.release(value.id, slot), Ok(()));
        assert_eq!(
            pool.release(value.id, slot),
            Err(ExecutionError::ValueUnavailable)
        );

        let lease = pool.acquire(slot, NodeId(1), value).unwrap();
        pool.commit(lease, value).unwrap();
        pool.pin(value.id, slot).unwrap();
        assert_eq!(
            pool.release(value.id, slot),
            Err(ExecutionError::ValueUnavailable)
        );
        assert_eq!(pool.deliver(value, slot).unwrap().len(), value.numel);
        assert!(matches!(
            pool.acquire(slot, NodeId(2), value),
            Err(ExecutionError::SlotUnavailable)
        ));
    }

    #[test]
    fn physical_capacity_allows_smaller_shapes_only_with_matching_metadata() {
        let (graph, _) = transformer_graph(2, 2, 2);
        let mut storage = plan_storage(&graph).unwrap();
        let value = &graph.values[graph.nodes[0].output.0];
        let slot = storage.assignments[value.id.0].unwrap();
        storage.slots[slot.0].capacity_bytes = value.bytes * 2;
        let mut pool = PhysicalStoragePool::new(&storage).unwrap();

        let lease = pool.acquire(slot, NodeId(0), value).unwrap();
        pool.abort(lease).unwrap();

        let mut smaller = value.clone();
        smaller.shape = vec![1, 2];
        smaller.numel = 2;
        smaller.bytes = 2 * size_of::<f32>();
        let lease = pool.acquire(slot, NodeId(1), &smaller).unwrap();
        pool.abort(lease).unwrap();

        let mut wrong_dtype = smaller.clone();
        wrong_dtype.dtype = GraphDType::Float64;
        assert!(matches!(
            pool.acquire(slot, NodeId(2), &wrong_dtype),
            Err(ExecutionError::IncompatibleDType)
        ));
        let mut wrong_layout = smaller;
        wrong_layout.layout = Layout::ContiguousColumnMajor;
        assert!(matches!(
            pool.acquire(slot, NodeId(2), &wrong_layout),
            Err(ExecutionError::IncompatibleLayout)
        ));
    }

    #[test]
    fn controlled_kernel_error_and_panic_do_not_poison_later_execution() {
        let (graph, [a_id, b_id, residual_id]) = transformer_graph(2, 2, 2);
        let storage = plan_storage(&graph).unwrap();
        let plan = ExecutionPlan::new(&graph, &storage).unwrap();
        let data = [1.0; 4];
        let mut external = ExternalInputs::new(graph.values.len());
        for id in [a_id, b_id, residual_id] {
            external
                .bind(
                    id,
                    ExternalTensor {
                        dtype: GraphDType::Float32,
                        layout: Layout::ContiguousRowMajor,
                        shape: &[2, 2],
                        data: &data,
                    },
                )
                .unwrap();
        }

        FAILURE_INJECTION.with(|injection| injection.set(Some(FailureInjection::Error)));
        assert_eq!(
            execute(&graph, &storage, &plan, &external),
            Err(ExecutionError::KernelFailure)
        );
        FAILURE_INJECTION.with(|injection| injection.set(Some(FailureInjection::Panic)));
        assert_eq!(
            execute(&graph, &storage, &plan, &external),
            Err(ExecutionError::KernelPanic)
        );
        assert!(execute(&graph, &storage, &plan, &external).is_ok());
    }

    #[test]
    fn repeated_execution_is_deterministic_and_reports_per_run_allocations() {
        let (m, k, n) = (2, 3, 2);
        let a = [0.0, -1.0, 1.0e-6, 1.0e3, -1.0e-3, 2.0];
        let b = [1.0, -2.0, 0.0, 3.0, -4.0, 1.0e-4];
        let residual = [-1.0e2, 1.0e2, -0.5, 0.5];
        let expected = execute_transformer(m, k, n, &a, &b, &residual);
        for _ in 0..32 {
            let actual = execute_transformer(m, k, n, &a, &b, &residual);
            assert_eq!(actual.outputs, expected.outputs);
            assert_eq!(actual.metrics, expected.metrics);
            assert_eq!(actual.metrics.physical_allocations, 2);
        }
    }

    #[test]
    fn branching_executes_without_releasing_shared_input_early() {
        let mut builder = GraphBuilder::new();
        let input_id = input(&mut builder, &[2, 2]);
        let softmax = builder
            .operation(Operation::SoftmaxLastDim, &[input_id])
            .unwrap();
        let transpose = builder
            .operation(Operation::Transpose, &[input_id])
            .unwrap();
        let output = builder
            .operation(Operation::Add, &[softmax, transpose])
            .unwrap();
        builder.mark_external_output(output).unwrap();
        let mut graph = builder.build();
        graph.analyze_lifetimes();
        let storage = plan_storage(&graph).unwrap();
        let plan = ExecutionPlan::new(&graph, &storage).unwrap();
        let data = [1.0, 2.0, 3.0, 4.0];
        let mut external = ExternalInputs::new(graph.values.len());
        external
            .bind(
                input_id,
                ExternalTensor {
                    dtype: GraphDType::Float32,
                    layout: Layout::ContiguousRowMajor,
                    shape: &[2, 2],
                    data: &data,
                },
            )
            .unwrap();
        let result = execute(&graph, &storage, &plan, &external).unwrap();
        let mut expected_softmax = [0.0; 4];
        let mut expected_transpose = [0.0; 4];
        let mut expected = [0.0; 4];
        softmax_last_dim_f32(&data, &mut expected_softmax, 2).unwrap();
        transpose_f32(&data, &mut expected_transpose, 2, 2).unwrap();
        add_f32(&expected_softmax, &expected_transpose, &mut expected).unwrap();
        assert_eq!(result.outputs[0].data, expected);
        assert_eq!(graph.values[input_id.0].last_use, Some(NodeId(1)));
    }

    #[test]
    fn intermediate_external_output_remains_available_to_later_consumer() {
        let mut builder = GraphBuilder::new();
        let input_id = input(&mut builder, &[2, 2]);
        let intermediate = builder
            .operation(Operation::SoftmaxLastDim, &[input_id])
            .unwrap();
        builder.mark_external_output(intermediate).unwrap();
        let final_output = builder
            .operation(Operation::Transpose, &[intermediate])
            .unwrap();
        builder.mark_external_output(final_output).unwrap();
        let mut graph = builder.build();
        graph.analyze_lifetimes();
        let storage = plan_storage(&graph).unwrap();
        let plan = ExecutionPlan::new(&graph, &storage).unwrap();
        let data = [1.0, 2.0, 3.0, 4.0];
        let mut external = ExternalInputs::new(graph.values.len());
        external
            .bind(
                input_id,
                ExternalTensor {
                    dtype: GraphDType::Float32,
                    layout: Layout::ContiguousRowMajor,
                    shape: &[2, 2],
                    data: &data,
                },
            )
            .unwrap();
        let result = execute(&graph, &storage, &plan, &external).unwrap();
        assert_eq!(result.outputs.len(), 2);
        assert_eq!(result.outputs[0].id, intermediate);
        assert_eq!(result.outputs[1].id, final_output);
        assert_eq!(result.metrics.releases, 0);
    }

    #[test]
    fn all_intermediates_can_be_external_outputs_without_early_release() {
        let mut builder = GraphBuilder::new();
        let input_id = input(&mut builder, &[2, 2]);
        let first = builder
            .operation(Operation::SoftmaxLastDim, &[input_id])
            .unwrap();
        let second = builder.operation(Operation::Transpose, &[first]).unwrap();
        let third = builder.operation(Operation::Transpose, &[second]).unwrap();
        for value in [first, second, third] {
            builder.mark_external_output(value).unwrap();
        }
        let mut graph = builder.build();
        graph.analyze_lifetimes();
        let storage = plan_storage(&graph).unwrap();
        let plan = ExecutionPlan::new(&graph, &storage).unwrap();
        let data = [0.0, -1.0, 100.0, -100.0];
        let mut external = ExternalInputs::new(graph.values.len());
        external
            .bind(
                input_id,
                ExternalTensor {
                    dtype: GraphDType::Float32,
                    layout: Layout::ContiguousRowMajor,
                    shape: &[2, 2],
                    data: &data,
                },
            )
            .unwrap();
        let result = execute(&graph, &storage, &plan, &external).unwrap();
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.id)
                .collect::<Vec<_>>(),
            vec![first, second, third]
        );
        assert_eq!(result.metrics.releases, 0);
    }
}
