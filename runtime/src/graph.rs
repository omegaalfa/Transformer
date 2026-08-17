//! Experimental graph lifetime and storage planning infrastructure.
//!
//! This module is deliberately disconnected from the FFI and kernel execution
//! paths. It proves storage liveness from a topologically ordered DAG; it never
//! infers ownership from native or PHP reference counts.

#[path = "graph/execution.rs"]
mod execution;

#[allow(unused_imports)]
pub(crate) use execution::{
    execute, ExecutionError, ExecutionPlan, ExecutionResult, ExternalInputs, ExternalTensor,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct ValueId(pub(crate) usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct NodeId(pub(crate) usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct StorageSlotId(pub(crate) usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum GraphDType {
    Float32,
    Float64,
}

impl GraphDType {
    fn element_size(self) -> usize {
        match self {
            Self::Float32 => size_of::<f32>(),
            Self::Float64 => size_of::<f64>(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Layout {
    ContiguousRowMajor,
    ContiguousColumnMajor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Operation {
    Matmul,
    Add,
    SoftmaxLastDim,
    Transpose,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Lifetime {
    pub(crate) start: NodeId,
    pub(crate) end: NodeId,
}

impl Lifetime {
    fn overlaps(self, other: Self) -> bool {
        !(self.end < other.start || other.end < self.start)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Value {
    pub(crate) id: ValueId,
    pub(crate) dtype: GraphDType,
    pub(crate) shape: Vec<usize>,
    pub(crate) numel: usize,
    pub(crate) bytes: usize,
    pub(crate) layout: Layout,
    pub(crate) external_input: bool,
    pub(crate) external_output: bool,
    pub(crate) producer: Option<NodeId>,
    pub(crate) consumers: Vec<NodeId>,
    pub(crate) consumer_count: usize,
    pub(crate) last_use: Option<NodeId>,
    pub(crate) lifetime: Option<Lifetime>,
    pub(crate) recyclable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Node {
    pub(crate) id: NodeId,
    pub(crate) operation: Operation,
    pub(crate) inputs: Vec<ValueId>,
    pub(crate) output: ValueId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Graph {
    pub(crate) values: Vec<Value>,
    pub(crate) nodes: Vec<Node>,
    analyzed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GraphError {
    InvalidGraph,
    InvalidValue,
    InvalidShape,
    IncompatibleShape,
    IncompatibleDType,
    IncompatibleLayout,
    ElementCountOverflow,
    GraphNotAnalyzed,
}

#[derive(Debug, Default)]
pub(crate) struct GraphBuilder {
    values: Vec<Value>,
    nodes: Vec<Node>,
}

impl GraphBuilder {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn input(
        &mut self,
        dtype: GraphDType,
        shape: Vec<usize>,
        layout: Layout,
    ) -> Result<ValueId, GraphError> {
        self.push_value(dtype, shape, layout, true, None)
    }

    pub(crate) fn operation(
        &mut self,
        operation: Operation,
        inputs: &[ValueId],
    ) -> Result<ValueId, GraphError> {
        let metadata = self.infer_output(operation, inputs)?;
        let node = NodeId(self.nodes.len());
        let output = self.push_value(metadata.0, metadata.1, metadata.2, false, Some(node))?;
        self.nodes.push(Node {
            id: node,
            operation,
            inputs: inputs.to_vec(),
            output,
        });
        Ok(output)
    }

    pub(crate) fn mark_external_output(&mut self, value: ValueId) -> Result<(), GraphError> {
        let value = self
            .values
            .get_mut(value.0)
            .ok_or(GraphError::InvalidValue)?;
        value.external_output = true;
        Ok(())
    }

    pub(crate) fn build(self) -> Graph {
        Graph {
            values: self.values,
            nodes: self.nodes,
            analyzed: false,
        }
    }

    fn push_value(
        &mut self,
        dtype: GraphDType,
        shape: Vec<usize>,
        layout: Layout,
        external_input: bool,
        producer: Option<NodeId>,
    ) -> Result<ValueId, GraphError> {
        let numel = checked_numel(&shape)?;
        let bytes = numel
            .checked_mul(dtype.element_size())
            .ok_or(GraphError::ElementCountOverflow)?;
        let id = ValueId(self.values.len());
        self.values.push(Value {
            id,
            dtype,
            shape,
            numel,
            bytes,
            layout,
            external_input,
            external_output: false,
            producer,
            consumers: Vec::new(),
            consumer_count: 0,
            last_use: None,
            lifetime: None,
            recyclable: false,
        });
        Ok(id)
    }

    fn infer_output(
        &self,
        operation: Operation,
        inputs: &[ValueId],
    ) -> Result<(GraphDType, Vec<usize>, Layout), GraphError> {
        let values = inputs
            .iter()
            .map(|id| self.values.get(id.0).ok_or(GraphError::InvalidValue))
            .collect::<Result<Vec<_>, _>>()?;
        match (operation, values.as_slice()) {
            (Operation::Matmul, [left, right]) => {
                compatible_metadata(left, right)?;
                if left.shape.len() != 2
                    || right.shape.len() != 2
                    || left.shape[1] != right.shape[0]
                {
                    return Err(GraphError::IncompatibleShape);
                }
                Ok((left.dtype, vec![left.shape[0], right.shape[1]], left.layout))
            }
            (Operation::Add, [left, right]) => {
                compatible_metadata(left, right)?;
                if left.shape != right.shape {
                    return Err(GraphError::IncompatibleShape);
                }
                Ok((left.dtype, left.shape.clone(), left.layout))
            }
            (Operation::SoftmaxLastDim, [input]) => {
                if input.shape.last().copied().unwrap_or(0) == 0 {
                    return Err(GraphError::InvalidShape);
                }
                Ok((input.dtype, input.shape.clone(), input.layout))
            }
            (Operation::Transpose, [input]) => {
                if input.shape.len() != 2 {
                    return Err(GraphError::InvalidShape);
                }
                Ok((
                    input.dtype,
                    vec![input.shape[1], input.shape[0]],
                    input.layout,
                ))
            }
            _ => Err(GraphError::InvalidShape),
        }
    }
}

impl Graph {
    pub(crate) fn analyze_lifetimes(&mut self) {
        self.try_analyze_lifetimes()
            .expect("GraphBuilder must produce a valid topological graph");
    }

    pub(crate) fn try_analyze_lifetimes(&mut self) -> Result<(), GraphError> {
        for (index, node) in self.nodes.iter().enumerate() {
            if node.id != NodeId(index) {
                return Err(GraphError::InvalidGraph);
            }
            let output = self
                .values
                .get(node.output.0)
                .ok_or(GraphError::InvalidGraph)?;
            if output.producer != Some(node.id) {
                return Err(GraphError::InvalidGraph);
            }
            for input in &node.inputs {
                let input = self.values.get(input.0).ok_or(GraphError::InvalidGraph)?;
                if input.producer.is_some_and(|producer| producer >= node.id) {
                    return Err(GraphError::InvalidGraph);
                }
            }
        }
        for value in &mut self.values {
            value.consumers.clear();
        }
        for node in &self.nodes {
            for input in &node.inputs {
                self.values
                    .get_mut(input.0)
                    .ok_or(GraphError::InvalidGraph)?
                    .consumers
                    .push(node.id);
            }
        }
        let graph_end = NodeId(self.nodes.len());
        for value in &mut self.values {
            value.consumer_count = value.consumers.len();
            value.last_use = value.consumers.last().copied();
            value.lifetime = value.producer.map(|start| Lifetime {
                start,
                end: if value.external_output {
                    graph_end
                } else {
                    value.last_use.unwrap_or(start)
                },
            });
            value.recyclable =
                value.producer.is_some() && !value.external_input && !value.external_output;
        }
        self.analyzed = true;
        Ok(())
    }

    pub(crate) fn is_analyzed(&self) -> bool {
        self.analyzed
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StorageSlot {
    pub(crate) id: StorageSlotId,
    pub(crate) dtype: GraphDType,
    pub(crate) layout: Layout,
    pub(crate) capacity_bytes: usize,
    pub(crate) values: Vec<ValueId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StoragePlan {
    pub(crate) slots: Vec<StorageSlot>,
    pub(crate) assignments: Vec<Option<StorageSlotId>>,
}

pub(crate) fn plan_storage(graph: &Graph) -> Result<StoragePlan, GraphError> {
    if !graph.is_analyzed() {
        return Err(GraphError::GraphNotAnalyzed);
    }
    let mut slots: Vec<StorageSlot> = Vec::new();
    let mut assignments = vec![None; graph.values.len()];

    for value in graph.values.iter().filter(|value| value.producer.is_some()) {
        let lifetime = value.lifetime.ok_or(GraphError::InvalidGraph)?;
        let reusable_slot = slots.iter_mut().find(|slot| {
            slot.dtype == value.dtype
                && slot.layout == value.layout
                && slot.capacity_bytes >= value.bytes
                && slot.values.iter().all(|assigned| {
                    let Some(other) = graph
                        .values
                        .get(assigned.0)
                        .and_then(|value| value.lifetime)
                    else {
                        return false;
                    };
                    !lifetime.overlaps(other)
                })
        });
        let slot_id = if let Some(slot) = reusable_slot {
            slot.values.push(value.id);
            slot.id
        } else {
            let id = StorageSlotId(slots.len());
            slots.push(StorageSlot {
                id,
                dtype: value.dtype,
                layout: value.layout,
                capacity_bytes: value.bytes,
                values: vec![value.id],
            });
            id
        };
        assignments[value.id.0] = Some(slot_id);
    }

    Ok(StoragePlan { slots, assignments })
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct PoolMetrics {
    pub(crate) physical_allocations: usize,
    pub(crate) reuses: usize,
    pub(crate) releases: usize,
    pub(crate) bytes_allocated: usize,
    pub(crate) bytes_reused: usize,
    pub(crate) peak_live_bytes: usize,
    pub(crate) peak_live_slots: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Simulation {
    pub(crate) metrics: PoolMetrics,
    pub(crate) events: Vec<String>,
}

struct StoragePool<'a> {
    plan: &'a StoragePlan,
    allocated: Vec<bool>,
    active: Vec<bool>,
    live_bytes: usize,
    live_slots: usize,
    metrics: PoolMetrics,
}

impl<'a> StoragePool<'a> {
    fn new(plan: &'a StoragePlan) -> Self {
        Self {
            plan,
            allocated: vec![false; plan.slots.len()],
            active: vec![false; plan.slots.len()],
            live_bytes: 0,
            live_slots: 0,
            metrics: PoolMetrics::default(),
        }
    }

    fn acquire(&mut self, slot_id: StorageSlotId, value_bytes: usize) -> bool {
        let slot = &self.plan.slots[slot_id.0];
        let reused = self.allocated[slot_id.0];
        if reused {
            self.metrics.reuses += 1;
            self.metrics.bytes_reused += value_bytes;
        } else {
            self.allocated[slot_id.0] = true;
            self.metrics.physical_allocations += 1;
            self.metrics.bytes_allocated += slot.capacity_bytes;
        }
        if !self.active[slot_id.0] {
            self.active[slot_id.0] = true;
            self.live_bytes += slot.capacity_bytes;
            self.live_slots += 1;
            self.metrics.peak_live_bytes = self.metrics.peak_live_bytes.max(self.live_bytes);
            self.metrics.peak_live_slots = self.metrics.peak_live_slots.max(self.live_slots);
        }
        reused
    }

    fn release(&mut self, slot_id: StorageSlotId) -> bool {
        if !self.active[slot_id.0] {
            return false;
        }
        self.active[slot_id.0] = false;
        self.live_bytes -= self.plan.slots[slot_id.0].capacity_bytes;
        self.live_slots -= 1;
        self.metrics.releases += 1;
        true
    }
}

pub(crate) fn simulate(graph: &Graph, plan: &StoragePlan) -> Result<Simulation, GraphError> {
    if !graph.is_analyzed() {
        return Err(GraphError::GraphNotAnalyzed);
    }
    if plan.assignments.len() != graph.values.len()
        || plan
            .slots
            .iter()
            .enumerate()
            .any(|(index, slot)| slot.id != StorageSlotId(index))
        || graph
            .values
            .iter()
            .filter(|value| value.producer.is_some())
            .any(|value| {
                plan.assignments
                    .get(value.id.0)
                    .copied()
                    .flatten()
                    .map_or(true, |slot| slot.0 >= plan.slots.len())
            })
    {
        return Err(GraphError::InvalidGraph);
    }
    let mut pool = StoragePool::new(plan);
    let mut events = Vec::new();

    for node in &graph.nodes {
        let value = &graph.values[node.output.0];
        let slot_id = plan.assignments[value.id.0].ok_or(GraphError::InvalidGraph)?;
        if pool.acquire(slot_id, value.bytes) {
            events.push(format!("node {} reuse slot {}", node.id.0, slot_id.0));
        } else {
            events.push(format!("node {} allocate slot {}", node.id.0, slot_id.0));
        }

        for input in &node.inputs {
            let input_value = &graph.values[input.0];
            if input_value.external_input
                || input_value.external_output
                || input_value.last_use != Some(node.id)
            {
                continue;
            }
            let Some(input_slot) = plan.assignments[input.0] else {
                continue;
            };
            if pool.release(input_slot) {
                events.push(format!("node {} release value {}", node.id.0, input.0));
            }
        }
    }
    Ok(Simulation {
        metrics: pool.metrics,
        events,
    })
}

fn checked_numel(shape: &[usize]) -> Result<usize, GraphError> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape.iter().try_fold(1usize, |total, dimension| {
        total
            .checked_mul(*dimension)
            .ok_or(GraphError::ElementCountOverflow)
    })
}

fn compatible_metadata(left: &Value, right: &Value) -> Result<(), GraphError> {
    if left.dtype != right.dtype {
        return Err(GraphError::IncompatibleDType);
    }
    if left.layout != right.layout {
        return Err(GraphError::IncompatibleLayout);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(builder: &mut GraphBuilder, shape: &[usize]) -> ValueId {
        builder
            .input(
                GraphDType::Float32,
                shape.to_vec(),
                Layout::ContiguousRowMajor,
            )
            .unwrap()
    }

    fn linear_graph() -> (Graph, [ValueId; 4]) {
        let mut builder = GraphBuilder::new();
        let a = input(&mut builder, &[128, 768]);
        let b = input(&mut builder, &[768, 768]);
        let residual = input(&mut builder, &[128, 768]);
        let x = builder.operation(Operation::Matmul, &[a, b]).unwrap();
        let y = builder.operation(Operation::Add, &[x, residual]).unwrap();
        let z = builder.operation(Operation::SoftmaxLastDim, &[y]).unwrap();
        let output = builder.operation(Operation::Transpose, &[z]).unwrap();
        builder.mark_external_output(output).unwrap();
        let mut graph = builder.build();
        graph.analyze_lifetimes();
        (graph, [x, y, z, output])
    }

    #[test]
    fn computes_consumers_last_use_and_topological_order() {
        let (graph, [x, y, z, output]) = linear_graph();
        assert_eq!(
            graph.nodes.iter().map(|node| node.id).collect::<Vec<_>>(),
            vec![NodeId(0), NodeId(1), NodeId(2), NodeId(3)]
        );
        assert_eq!(graph.values[x.0].last_use, Some(NodeId(1)));
        assert_eq!(graph.values[y.0].last_use, Some(NodeId(2)));
        assert_eq!(graph.values[z.0].last_use, Some(NodeId(3)));
        assert_eq!(graph.values[output.0].consumer_count, 0);
        assert!(!graph.values[output.0].recyclable);
    }

    #[test]
    fn linear_pipeline_uses_two_slots_and_reuses_each_once() {
        let (graph, [x, y, z, output]) = linear_graph();
        let plan = plan_storage(&graph).unwrap();
        assert_eq!(plan.slots.len(), 2);
        assert_eq!(plan.assignments[x.0], plan.assignments[z.0]);
        assert_eq!(plan.assignments[y.0], plan.assignments[output.0]);
        let simulation = simulate(&graph, &plan).unwrap();
        assert_eq!(simulation.metrics.physical_allocations, 2);
        assert_eq!(simulation.metrics.reuses, 2);
        assert_eq!(simulation.metrics.peak_live_bytes, 2 * 128 * 768 * 4);
    }

    #[test]
    fn branching_keeps_value_alive_until_its_last_consumer() {
        let mut builder = GraphBuilder::new();
        let left = input(&mut builder, &[2, 2]);
        let right = input(&mut builder, &[2, 2]);
        let x = builder
            .operation(Operation::Matmul, &[left, right])
            .unwrap();
        let first = builder.operation(Operation::SoftmaxLastDim, &[x]).unwrap();
        let second = builder.operation(Operation::Transpose, &[x]).unwrap();
        let output = builder.operation(Operation::Add, &[first, second]).unwrap();
        builder.mark_external_output(output).unwrap();
        let mut graph = builder.build();
        graph.analyze_lifetimes();
        assert_eq!(graph.values[x.0].consumer_count, 2);
        assert_eq!(graph.values[x.0].last_use, Some(NodeId(2)));
        let plan = plan_storage(&graph).unwrap();
        assert_ne!(plan.assignments[x.0], plan.assignments[first.0]);
        assert_ne!(plan.assignments[x.0], plan.assignments[second.0]);
        assert_eq!(plan.assignments[x.0], plan.assignments[output.0]);
    }

    #[test]
    fn external_inputs_have_no_pool_slot_and_output_is_pinned() {
        let (graph, [x, _, _, output]) = linear_graph();
        for value in graph.values.iter().filter(|value| value.external_input) {
            assert!(value.lifetime.is_none());
            assert!(!value.recyclable);
        }
        assert!(graph.values[output.0].external_output);
        assert_eq!(graph.values[output.0].lifetime.unwrap().end, NodeId(4));
        assert!(graph.values[x.0].recyclable);
    }

    #[test]
    fn external_intermediate_is_not_recyclable_or_overwritten() {
        let mut builder = GraphBuilder::new();
        let a = input(&mut builder, &[2, 2]);
        let b = input(&mut builder, &[2, 2]);
        let x = builder.operation(Operation::Matmul, &[a, b]).unwrap();
        builder.mark_external_output(x).unwrap();
        let _y = builder.operation(Operation::SoftmaxLastDim, &[x]).unwrap();
        let mut graph = builder.build();
        graph.analyze_lifetimes();
        let plan = plan_storage(&graph).unwrap();
        assert!(!graph.values[x.0].recyclable);
        assert_ne!(plan.assignments[x.0], plan.assignments[_y.0]);
        let simulation = simulate(&graph, &plan).unwrap();
        assert!(!simulation
            .events
            .iter()
            .any(|event| event.ends_with(&format!("release value {}", x.0))));
    }

    #[test]
    fn rejects_incompatible_shapes_dtype_and_layout() {
        let mut shapes = GraphBuilder::new();
        let a = input(&mut shapes, &[2, 3]);
        let b = input(&mut shapes, &[4, 2]);
        assert_eq!(
            shapes.operation(Operation::Matmul, &[a, b]),
            Err(GraphError::IncompatibleShape)
        );

        let mut dtype = GraphBuilder::new();
        let f32_value = input(&mut dtype, &[2, 2]);
        let f64_value = dtype
            .input(GraphDType::Float64, vec![2, 2], Layout::ContiguousRowMajor)
            .unwrap();
        assert_eq!(
            dtype.operation(Operation::Add, &[f32_value, f64_value]),
            Err(GraphError::IncompatibleDType)
        );

        let mut layout = GraphBuilder::new();
        let row = input(&mut layout, &[2, 2]);
        let column = layout
            .input(
                GraphDType::Float32,
                vec![2, 2],
                Layout::ContiguousColumnMajor,
            )
            .unwrap();
        assert_eq!(
            layout.operation(Operation::Add, &[row, column]),
            Err(GraphError::IncompatibleLayout)
        );
    }

    #[test]
    fn planner_requires_analysis_and_respects_overlapping_intervals() {
        let mut builder = GraphBuilder::new();
        let a = input(&mut builder, &[2, 2]);
        let b = input(&mut builder, &[2, 2]);
        let x = builder.operation(Operation::Matmul, &[a, b]).unwrap();
        let y = builder.operation(Operation::SoftmaxLastDim, &[x]).unwrap();
        let graph = builder.build();
        assert_eq!(plan_storage(&graph), Err(GraphError::GraphNotAnalyzed));
        let mut graph = graph;
        graph.analyze_lifetimes();
        let plan = plan_storage(&graph).unwrap();
        assert_ne!(plan.assignments[x.0], plan.assignments[y.0]);
    }

    #[test]
    fn larger_value_does_not_reuse_an_incompatible_capacity() {
        let mut builder = GraphBuilder::new();
        let small_a = input(&mut builder, &[1, 2]);
        let small_b = input(&mut builder, &[2, 1]);
        let small = builder
            .operation(Operation::Matmul, &[small_a, small_b])
            .unwrap();
        let _small_use = builder
            .operation(Operation::SoftmaxLastDim, &[small])
            .unwrap();
        let large_a = input(&mut builder, &[4, 4]);
        let large_b = input(&mut builder, &[4, 4]);
        let large = builder
            .operation(Operation::Matmul, &[large_a, large_b])
            .unwrap();
        let mut graph = builder.build();
        graph.analyze_lifetimes();
        let plan = plan_storage(&graph).unwrap();
        assert_ne!(plan.assignments[small.0], plan.assignments[large.0]);
    }

    #[test]
    fn rejects_malformed_graphs_and_storage_plans_without_panicking() {
        let (mut graph, _) = linear_graph();
        graph.nodes[0].id = NodeId(7);
        assert_eq!(graph.try_analyze_lifetimes(), Err(GraphError::InvalidGraph));

        let (graph, _) = linear_graph();
        let mut plan = plan_storage(&graph).unwrap();
        plan.assignments.pop();
        assert_eq!(simulate(&graph, &plan), Err(GraphError::InvalidGraph));
    }

    #[test]
    fn analysis_planning_and_simulation_are_deterministic() {
        let (expected_graph, _) = linear_graph();
        let expected_plan = plan_storage(&expected_graph).unwrap();
        let expected_simulation = simulate(&expected_graph, &expected_plan).unwrap();
        for _ in 0..100 {
            let (graph, _) = linear_graph();
            let plan = plan_storage(&graph).unwrap();
            assert_eq!(graph, expected_graph);
            assert_eq!(plan, expected_plan);
            assert_eq!(simulate(&graph, &plan).unwrap(), expected_simulation);
        }
    }

    #[test]
    fn lifetime_patterns_never_reuse_before_the_last_consumer() {
        // C: a produced X feeds three independently ordered consumers.
        let mut builder = GraphBuilder::new();
        let source = input(&mut builder, &[2, 2]);
        let x = builder
            .operation(Operation::SoftmaxLastDim, &[source])
            .unwrap();
        let a = builder.operation(Operation::SoftmaxLastDim, &[x]).unwrap();
        let b = builder.operation(Operation::Transpose, &[x]).unwrap();
        let c = builder.operation(Operation::Transpose, &[x]).unwrap();
        let mut graph = builder.build();
        graph.analyze_lifetimes();
        let plan = plan_storage(&graph).unwrap();
        assert_eq!(graph.values[x.0].producer, Some(NodeId(0)));
        assert_eq!(
            graph.values[x.0].consumers,
            vec![NodeId(1), NodeId(2), NodeId(3)]
        );
        assert_eq!(graph.values[x.0].consumer_count, 3);
        assert_eq!(graph.values[x.0].last_use, Some(NodeId(3)));
        assert_eq!(
            graph.values[x.0].lifetime,
            Some(Lifetime {
                start: NodeId(0),
                end: NodeId(3)
            })
        );
        for consumer in [a, b, c] {
            assert_ne!(plan.assignments[x.0], plan.assignments[consumer.0]);
        }

        // D: two independent sources converge at C.
        let mut builder = GraphBuilder::new();
        let x = input(&mut builder, &[2, 2]);
        let y = input(&mut builder, &[2, 2]);
        let a = builder.operation(Operation::SoftmaxLastDim, &[x]).unwrap();
        let b = builder.operation(Operation::SoftmaxLastDim, &[y]).unwrap();
        let c = builder.operation(Operation::Add, &[a, b]).unwrap();
        let mut graph = builder.build();
        graph.analyze_lifetimes();
        let plan = plan_storage(&graph).unwrap();
        assert_eq!(graph.values[a.0].consumers, vec![NodeId(2)]);
        assert_eq!(graph.values[b.0].consumers, vec![NodeId(2)]);
        assert_eq!(graph.values[a.0].last_use, Some(NodeId(2)));
        assert_eq!(graph.values[b.0].last_use, Some(NodeId(2)));
        assert_ne!(plan.assignments[a.0], plan.assignments[c.0]);
        assert_ne!(plan.assignments[b.0], plan.assignments[c.0]);

        // E: Y participates in A and a later independent C while A feeds B.
        let mut builder = GraphBuilder::new();
        let x = input(&mut builder, &[2, 2]);
        let y = input(&mut builder, &[2, 2]);
        let a = builder.operation(Operation::Add, &[x, y]).unwrap();
        let b = builder.operation(Operation::SoftmaxLastDim, &[a]).unwrap();
        let c = builder.operation(Operation::Add, &[y, y]).unwrap();
        let mut graph = builder.build();
        graph.analyze_lifetimes();
        let plan = plan_storage(&graph).unwrap();
        assert_eq!(
            graph.values[y.0].consumers,
            vec![NodeId(0), NodeId(2), NodeId(2)]
        );
        assert_eq!(graph.values[y.0].consumer_count, 3);
        assert_eq!(graph.values[y.0].last_use, Some(NodeId(2)));
        assert!(plan.assignments[y.0].is_none());
        assert_eq!(graph.values[a.0].last_use, Some(NodeId(1)));
        assert_ne!(plan.assignments[a.0], plan.assignments[b.0]);
        assert_eq!(plan.assignments[a.0], plan.assignments[c.0]);
    }
}
