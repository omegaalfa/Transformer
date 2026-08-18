<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Backend\Cuda;

use FFI;
use FFI\CData;
use Omegaalfa\Transformer\Exception\BackendException;

final class CudaBgeLibrary
{
    private FFI $ffi;
    private mixed $handle;
    private int $parameterCount = 0;

    public function __construct(string $libraryPath, public readonly CudaBgePrecision $precision = CudaBgePrecision::Float32)
    {
        $this->ffi = FFI::cdef(<<<'CDEF'
            int transformer_cuda_available(void);
            void *transformer_cuda_bge_create(void);
            void *transformer_cuda_bge_create_precision(int precision);
            int transformer_cuda_bge_set_parameter(void *handle, int index, const float *data, size_t count);
            int transformer_cuda_bge_set_parameter_bytes(void *handle, int index, const uint8_t *data,
                size_t byte_count, int rows, int columns, int transpose);
            int transformer_cuda_bge_finalize(void *handle);
            int transformer_cuda_bge_set_math_mode(void *handle, int mode);
            int transformer_cuda_bge_set_graph_enabled(void *handle, int enabled);
            int transformer_cuda_bge_forward(void *handle, const int64_t *ids, const uint8_t *mask,
                const int64_t *types, int batch, int sequence, float *output);
            int transformer_cuda_bge_profile(void *handle, const int64_t *ids, const uint8_t *mask,
                const int64_t *types, int batch, int sequence, float *output, float *timings,
                int timing_capacity, int *timing_count);
            int transformer_cuda_bge_forward_detailed(void *handle, const int64_t *ids, const uint8_t *mask,
                const int64_t *types, int batch, int sequence, float *output, float *phases);
            int transformer_cuda_bge_diagnose(void *handle, const int64_t *ids, const uint8_t *mask,
                const int64_t *types, int batch, int sequence, float *snapshots, size_t snapshot_count,
                uint64_t *metadata, float *invalid_value);
            int transformer_cuda_bge_diagnostics(void *handle, uint64_t *values, size_t capacity);
            void transformer_cuda_bge_destroy(void *handle);
            int transformer_cuda_memory_info(size_t *free_bytes, size_t *total_bytes);
            CDEF, $libraryPath);
        if ($this->invoke('transformer_cuda_available') !== 1) {
            throw new BackendException('CUDA device is unavailable.');
        }
        $handle = $this->invoke('transformer_cuda_bge_create_precision', $precision->value);
        if (!$handle instanceof CData || FFI::isNull($handle)) {
            throw new BackendException('CUDA BGE model allocation failed.');
        }
        $this->handle = $handle;
    }

    public function __destruct()
    {
        $this->destroy();
    }

    /** @param list<float> $values */
    public function setParameter(int $index, array $values): void
    {
        $buffer = $this->buffer('float[' . count($values) . ']');
        FFI::memcpy($buffer, pack('g*', ...$values), count($values) * 4);
        if ($this->invoke('transformer_cuda_bge_set_parameter', $this->handle(), $index, $buffer, count($values)) !== 0) {
            throw new BackendException("CUDA rejected BGE parameter {$index}.");
        }
        ++$this->parameterCount;
    }

    /** Internal checkpoint-loader boundary. The payload is F32 little-endian and never becomes PHP float zvals.
     * @param array{int, int} $checkpointShape
     */
    public function setParameterBytes(int $index, string $bytes, array $checkpointShape, bool $transpose): void
    {
        [$rows, $columns] = $checkpointShape;
        if ($rows < 1 || $columns < 1 || $rows > intdiv(PHP_INT_MAX, $columns)
            || $rows * $columns > intdiv(PHP_INT_MAX, 4) || strlen($bytes) !== $rows * $columns * 4) {
            throw new BackendException("CUDA parameter {$index} has invalid F32 payload dimensions.");
        }
        $buffer = $this->buffer('uint8_t[' . strlen($bytes) . ']');
        FFI::memcpy($buffer, $bytes, strlen($bytes));
        if ($this->invoke(
            'transformer_cuda_bge_set_parameter_bytes',
            $this->handle(),
            $index,
            $buffer,
            strlen($bytes),
            $rows,
            $columns,
            $transpose ? 1 : 0
        ) !== 0) {
            throw new BackendException("CUDA rejected binary BGE parameter {$index}.");
        }
        ++$this->parameterCount;
    }

    public function finalize(): void
    {
        if ($this->parameterCount !== 197 || $this->invoke('transformer_cuda_bge_finalize', $this->handle()) !== 0) {
            throw new BackendException('CUDA BGE model is incomplete.');
        }
    }

    public function setTensorFloat32(bool $enabled): void
    {
        if ($this->invoke('transformer_cuda_bge_set_math_mode', $this->handle(), $enabled ? 1 : 0) !== 0) {
            throw new BackendException('CUDA rejected the requested cuBLAS math mode.');
        }
    }

    /** Internal benchmark control; production keeps CUDA Graphs enabled. */
    public function setGraphEnabled(bool $enabled): void
    {
        if ($this->invoke('transformer_cuda_bge_set_graph_enabled', $this->handle(), $enabled ? 1 : 0) !== 0) {
            throw new BackendException('Failed to configure CUDA Graph execution.');
        }
    }

    /** @param list<int> $ids
     *  @param list<bool> $mask
     *  @param list<int> $types
     *  @return list<list<float>>
     */
    public function forward(array $ids, array $mask, array $types, int $batch, int $sequence): array
    {
        return $this->execute($ids, $mask, $types, $batch, $sequence, 0)['output'];
    }

    /** @param list<int> $ids
     *  @param list<bool> $mask
     *  @param list<int> $types
     *  @return array{output: list<list<float>>, timings_ms: list<float>}
     */
    public function profile(array $ids, array $mask, array $types, int $batch, int $sequence): array
    {
        return $this->execute($ids, $mask, $types, $batch, $sequence, 1);
    }

    /** Internal benchmark API with public-path preparation and materialization.
     *  @param list<int> $ids
     *  @param list<bool> $mask
     *  @param list<int> $types
     *  @return array{output: list<list<float>>, timings_ms: list<float>, phases_us: array<string, float>}
     */
    public function benchmarkForward(array $ids, array $mask, array $types, int $batch, int $sequence): array
    {
        $result = $this->execute($ids, $mask, $types, $batch, $sequence, 2);
        if (!isset($result['phases_us'])) {
            throw new BackendException('CUDA benchmark phases were not returned.');
        }

        return ['output' => $result['output'], 'timings_ms' => $result['timings_ms'], 'phases_us' => $result['phases_us']];
    }

    /** @return array<string, int|bool> */
    public function benchmarkDiagnostics(): array
    {
        $buffer = $this->buffer('uint64_t[25]');
        if ($this->invoke('transformer_cuda_bge_diagnostics', $this->handle(), $buffer, 25) !== 0) {
            throw new BackendException('CUDA BGE diagnostics failed.');
        }
        $decoded = unpack('P*', FFI::string($buffer, 25 * 8));
        if ($decoded === false || count($decoded) !== 25) {
            throw new BackendException('CUDA BGE diagnostics are malformed.');
        }
        $values = [];
        foreach ($decoded as $value) {
            if (!is_int($value)) {
                throw new BackendException('CUDA BGE diagnostic value is malformed.');
            }
            $values[] = $value;
        }

        return [
            'cuda_mallocs' => $values[0], 'cuda_frees' => $values[1],
            'host_submissions' => $values[2], 'parameter_uploads' => $values[3],
            'workspace_reallocations' => $values[4], 'h2d_bytes' => $values[5],
            'd2h_bytes' => $values[6], 'synchronizations' => $values[7],
            'graph_launches' => $values[8], 'internal_submissions' => $values[9],
            'graph_captures' => $values[10], 'graph_invalidations' => $values[11],
            'batch' => $values[12], 'sequence' => $values[13],
            'graph_enabled' => $values[14] === 1, 'graph_captured' => $values[15] === 1,
            'graph_reused' => $values[16] === 1, 'graph_ready' => $values[17] === 1,
            'parameter_bytes' => $values[18], 'workspace_bytes' => $values[19],
            'precision' => $values[20],
            'parameter_conversion_ns' => $values[21], 'parameter_upload_ns' => $values[22],
            'parameter_validation_ns' => $values[23], 'parameter_transpose_ns' => $values[24],
        ];
    }

    /** Internal mixed-precision validation; never used by public forward.
     * @param list<int> $ids
     * @param list<bool> $mask
     * @param list<int> $types
     * @return array{states: list<list<float>>, invalid: array{category: int, stage: int, layer: int, index: int, value: float}}
     */
    public function benchmarkStates(array $ids, array $mask, array $types, int $batch, int $sequence): array
    {
        $count = $batch * $sequence;
        if ($batch < 1 || $sequence < 1 || count($ids) !== $count || count($mask) !== $count || count($types) !== $count) {
            throw new BackendException('CUDA diagnostic input must match [B,S].');
        }
        $idBuffer = $this->buffer("int64_t[{$count}]");
        $maskBuffer = $this->buffer("uint8_t[{$count}]");
        $typeBuffer = $this->buffer("int64_t[{$count}]");
        $packedIds = pack('q*', ...$ids);
        $packedTypes = pack('q*', ...$types);
        $packedMask = pack('C*', ...array_map(static fn ($value): int => $value ? 1 : 0, $mask));
        FFI::memcpy($idBuffer, $packedIds, strlen($packedIds));
        FFI::memcpy($typeBuffer, $packedTypes, strlen($packedTypes));
        FFI::memcpy($maskBuffer, $packedMask, strlen($packedMask));
        $stateSize = $count * 384;
        $snapshotCount = 13 * $stateSize;
        $snapshots = $this->buffer("float[{$snapshotCount}]");
        $metadata = $this->buffer('uint64_t[4]');
        $invalidValue = $this->buffer('float[1]');
        if ($this->invoke('transformer_cuda_bge_diagnose', $this->handle(), $idBuffer, $maskBuffer, $typeBuffer, $batch, $sequence, $snapshots, $snapshotCount, $metadata, $invalidValue) !== 0) {
            throw new BackendException('CUDA mixed-precision diagnostic failed.');
        }
        $snapshotBytes = $this->checkedBytes($snapshotCount, 4);
        $decoded = unpack('g*', FFI::string($snapshots, $snapshotBytes));
        $decodedMetadata = unpack('P*', FFI::string($metadata, 32));
        $decodedInvalid = unpack('gvalue', FFI::string($invalidValue, 4));
        if ($decoded === false || $decodedMetadata === false || $decodedInvalid === false) {
            throw new BackendException('CUDA mixed-precision diagnostic is malformed.');
        }
        $values = [];
        foreach ($decoded as $value) {
            if (!is_float($value)) {
                throw new BackendException('CUDA mixed-precision snapshot is malformed.');
            }
            $values[] = $value;
        }
        $states = [];
        for ($state = 0; $state < 13; ++$state) {
            $states[] = array_slice($values, $state * $stateSize, $stateSize);
        }
        $meta = $this->decodeDiagnosticMetadata($decodedMetadata);
        $invalid = $decodedInvalid['value'];
        if (!is_float($invalid)) {
            throw new BackendException('CUDA mixed-precision invalid value is malformed.');
        }

        return ['states' => $states, 'invalid' => ['category' => $meta[0], 'stage' => $meta[1],
            'layer' => $meta[2] - 1, 'index' => $meta[3], 'value' => $invalid]];
    }

    /** @param list<int> $ids
     * @param list<bool> $mask
     * @param list<int> $types
     * @return array{category: int, stage: int, layer: int, index: int, value: float}
     */
    public function benchmarkFinite(array $ids, array $mask, array $types, int $batch, int $sequence): array
    {
        $count = $batch * $sequence;
        $idBuffer = $this->buffer("int64_t[{$count}]");
        $maskBuffer = $this->buffer("uint8_t[{$count}]");
        $typeBuffer = $this->buffer("int64_t[{$count}]");
        $packedIds = pack('q*', ...$ids);
        $packedTypes = pack('q*', ...$types);
        $packedMask = pack('C*', ...array_map(static fn ($value): int => $value ? 1 : 0, $mask));
        FFI::memcpy($idBuffer, $packedIds, strlen($packedIds));
        FFI::memcpy($typeBuffer, $packedTypes, strlen($packedTypes));
        FFI::memcpy($maskBuffer, $packedMask, strlen($packedMask));
        $metadata = $this->buffer('uint64_t[4]');
        $invalidValue = $this->buffer('float[1]');
        if ($this->invoke(
            'transformer_cuda_bge_diagnose',
            $this->handle(),
            $idBuffer,
            $maskBuffer,
            $typeBuffer,
            $batch,
            $sequence,
            $this->ffi->cast('float *', 0),
            0,
            $metadata,
            $invalidValue
        ) !== 0) {
            throw new BackendException('CUDA finite diagnostic failed.');
        }
        $decodedMetadata = unpack('P*', FFI::string($metadata, 32));
        $decodedInvalid = unpack('gvalue', FFI::string($invalidValue, 4));
        if ($decodedMetadata === false || $decodedInvalid === false) {
            throw new BackendException('CUDA finite diagnostic is malformed.');
        }
        $meta = $this->decodeDiagnosticMetadata($decodedMetadata);
        $invalid = $decodedInvalid['value'];
        if (!is_float($invalid)) {
            throw new BackendException('CUDA finite diagnostic value is malformed.');
        }

        return ['category' => $meta[0], 'stage' => $meta[1], 'layer' => $meta[2] - 1,
            'index' => $meta[3], 'value' => $invalid];
    }

    /** @param array<int|string, mixed> $decoded
     * @return list<int>
     */
    private function decodeDiagnosticMetadata(array $decoded): array
    {
        $metadata = [];
        foreach ($decoded as $value) {
            if (!is_int($value)) {
                throw new BackendException('CUDA diagnostic metadata is malformed.');
            }
            $metadata[] = $value;
        }
        if (count($metadata) !== 4) {
            throw new BackendException('CUDA diagnostic metadata is incomplete.');
        }

        return $metadata;
    }

    /** @return int<0, max> */
    private function checkedBytes(int $elements, int $elementSize): int
    {
        if ($elements < 0 || $elementSize < 0 || ($elementSize !== 0 && $elements > intdiv(PHP_INT_MAX, $elementSize))) {
            throw new BackendException('CUDA diagnostic byte size overflows PHP integer capacity.');
        }

        return $elements * $elementSize;
    }

    /** @param list<int> $ids
     *  @param list<bool> $mask
     *  @param list<int> $types
     *  @return array{output: list<list<float>>, timings_ms: list<float>, phases_us?: array<string, float>}
     */
    private function execute(array $ids, array $mask, array $types, int $batch, int $sequence, int $mode): array
    {
        $preparationStart = hrtime(true);
        if ($batch < 1 || $sequence < 1 || $sequence > 512) {
            throw new BackendException('CUDA BGE input must match [B,S] with 1 <= S <= 512.');
        }
        $count = $batch * $sequence;
        if (count($ids) !== $count || count($mask) !== $count || count($types) !== $count) {
            throw new BackendException('CUDA BGE input must match [B,S] with 1 <= S <= 512.');
        }
        foreach ($ids as $id) {
            if ($id < 0 || $id >= 30522) {
                throw new BackendException('CUDA BGE token ID is outside the checkpoint vocabulary.');
            }
        }
        foreach ($types as $type) {
            if ($type < 0 || $type >= 2) {
                throw new BackendException('CUDA BGE token type ID is outside the checkpoint vocabulary.');
            }
        }
        $idBuffer = $this->buffer("int64_t[{$count}]");
        $maskBuffer = $this->buffer("uint8_t[{$count}]");
        $typeBuffer = $this->buffer("int64_t[{$count}]");
        $integerBytes = $count * 8;
        FFI::memcpy($idBuffer, pack('q*', ...$ids), $integerBytes);
        FFI::memcpy($typeBuffer, pack('q*', ...$types), $integerBytes);
        FFI::memcpy($maskBuffer, pack('C*', ...array_map(static fn (bool $value): int => $value ? 1 : 0, $mask)), $count);
        $output = $this->buffer('float[' . ($batch * 384) . ']');
        $timings = $this->buffer('float[137]');
        $timingCount = $this->buffer('int[1]');
        $phases = $this->buffer('float[3]');
        $preparationUs = (hrtime(true) - $preparationStart) / 1_000;
        $ffiStart = hrtime(true);
        $status = match ($mode) {
            1 => $this->invoke('transformer_cuda_bge_profile', $this->handle(), $idBuffer, $maskBuffer, $typeBuffer, $batch, $sequence, $output, $timings, 137, $timingCount),
            2 => $this->invoke('transformer_cuda_bge_forward_detailed', $this->handle(), $idBuffer, $maskBuffer, $typeBuffer, $batch, $sequence, $output, $phases),
            default => $this->invoke('transformer_cuda_bge_forward', $this->handle(), $idBuffer, $maskBuffer, $typeBuffer, $batch, $sequence, $output),
        };
        $ffiUs = (hrtime(true) - $ffiStart) / 1_000;
        if ($status !== 0) {
            throw new BackendException('CUDA BGE forward failed.');
        }
        $materializationStart = hrtime(true);
        $decoded = unpack('g*', FFI::string($output, $batch * 384 * 4));
        if ($decoded === false) {
            throw new BackendException('CUDA embedding decoding failed.');
        }
        $values = [];
        foreach ($decoded as $value) {
            if (!is_float($value) || !is_finite($value)) {
                throw new BackendException('CUDA produced a non-finite embedding.');
            }
            $values[] = $value;
        }

        $rows = [];
        for ($row = 0; $row < $batch; ++$row) {
            $rows[] = array_slice($values, $row * 384, 384);
        }

        $timingValues = [];
        if ($mode === 1) {
            $countDecoded = unpack('lvalue', FFI::string($timingCount, 4));
            if ($countDecoded === false || !is_int($countDecoded['value']) || $countDecoded['value'] !== 137) {
                throw new BackendException('CUDA profiling result is malformed.');
            }
            $decodedTimings = unpack('g*', FFI::string($timings, 137 * 4));
            if ($decodedTimings === false) {
                throw new BackendException('CUDA profiling timings cannot be decoded.');
            }
            foreach ($decodedTimings as $timing) {
                if (!is_float($timing)) {
                    throw new BackendException('CUDA profiling timing is malformed.');
                }
                $timingValues[] = $timing;
            }
        }

        $result = ['output' => $rows, 'timings_ms' => $timingValues];
        if ($mode === 2) {
            $decodedPhases = unpack('g*', FFI::string($phases, 3 * 4));
            if ($decodedPhases === false || count($decodedPhases) !== 3) {
                throw new BackendException('CUDA phase timings are malformed.');
            }
            $phaseValues = [];
            foreach ($decodedPhases as $phase) {
                if (!is_float($phase)) {
                    throw new BackendException('CUDA phase timing value is malformed.');
                }
                $phaseValues[] = $phase;
            }
            $result['phases_us'] = [
                'preparation' => $preparationUs,
                'ffi_total' => $ffiUs,
                'h2d' => $phaseValues[0] * 1_000,
                'device' => $phaseValues[1] * 1_000,
                'd2h' => $phaseValues[2] * 1_000,
                'materialization' => (hrtime(true) - $materializationStart) / 1_000,
            ];
        }

        return $result;
    }

    public function parameterCount(): int
    {
        return $this->parameterCount;
    }

    public function identity(): int
    {
        return spl_object_id($this->handle());
    }

    /** @return array{free: int, total: int} */
    public function memoryInfo(): array
    {
        $free = $this->buffer('size_t[1]');
        $total = $this->buffer('size_t[1]');
        if ($this->invoke('transformer_cuda_memory_info', $free, $total) !== 0) {
            throw new BackendException('CUDA memory query failed.');
        }

        $freeValue = unpack('Pvalue', FFI::string($free, FFI::sizeof($free)));
        $totalValue = unpack('Pvalue', FFI::string($total, FFI::sizeof($total)));
        if (
            $freeValue === false
            || $totalValue === false
            || !is_int($freeValue['value'])
            || !is_int($totalValue['value'])
        ) {
            throw new BackendException('CUDA memory query decoding failed.');
        }

        return ['free' => $freeValue['value'], 'total' => $totalValue['value']];
    }

    public function destroy(): void
    {
        if ($this->handle === null) {
            return;
        }
        $this->invoke('transformer_cuda_bge_destroy', $this->handle);
        $this->handle = null;
    }

    private function handle(): CData
    {
        if (!$this->handle instanceof CData) {
            throw new BackendException('CUDA BGE model was destroyed.');
        }

        return $this->handle;
    }

    private function buffer(string $type): CData
    {
        return $this->ffi->new($type);
    }

    private function invoke(string $function, mixed ...$arguments): mixed
    {
        return $this->ffi->{$function}(...$arguments);
    }
}
