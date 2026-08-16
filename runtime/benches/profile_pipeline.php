<?php

declare(strict_types=1);

use FFI\CData;
use Omegaalfa\Transformer\Backend\Ffi\FfiBackend;
use Omegaalfa\Transformer\Backend\Ffi\NativeLibrary;

require dirname(__DIR__, 2) . '/vendor/autoload.php';

const M = 128;
const K = 768;
const N = 768;
const ROWS = 128;
const COLUMNS = 768;
const SOFTMAX_CALLS = 128;
const STATUS_OK = 0;

$samples = max(3, (int) (getenv('TRANSFORMER_PROFILE_SAMPLES') ?: 5));
$projectRoot = dirname(__DIR__, 2);
$libraryPath = NativeLibrary::defaultPath($projectRoot);
$definitions = file_get_contents($projectRoot . '/php/src/Backend/Ffi/definitions.h');

if ($definitions === false) {
    throw new RuntimeException('Unable to read FFI definitions.');
}

$ffi = FFI::cdef($definitions, $libraryPath);
$backend = new FfiBackend(new NativeLibrary($libraryPath));
$length = M * N;
$bLength = K * N;
$aValues = array_fill(0, M * K, 0.01);
$bValues = array_fill(0, $bLength, 0.02);
$residualValues = array_fill(0, $length, 0.03);
$vectorValues = array_fill(0, N, 0.01);

/** @var array<string, array{layer: string, calls: int, samples: list<float>}> $records */
$records = [];

/** @param callable(): void $operation */
function record(array &$records, string $layer, string $stage, int $calls, callable $operation): void
{
    $start = hrtime(true);
    $operation();
    $elapsed = (hrtime(true) - $start) / 1_000.0;
    $records[$layer . ':' . $stage] ??= ['layer' => $layer, 'calls' => $calls, 'samples' => []];
    $records[$layer . ':' . $stage]['samples'][] = $elapsed;
}

/** @param list<float> $values */
function fillBuffer(CData $buffer, array $values): void
{
    foreach ($values as $index => $value) {
        $buffer[$index] = $value;
    }
}

function requireStatus(int $status, string $operation): void
{
    if ($status !== STATUS_OK) {
        throw new RuntimeException("{$operation} failed with status {$status}.");
    }
}

/** @return list<float> */
function cdataToFloatList(CData $buffer, int $length): array
{
    $values = [];
    for ($index = 0; $index < $length; ++$index) {
        $value = $buffer[$index];
        if (!is_float($value)) {
            throw new RuntimeException('Native float buffer returned a non-float value.');
        }
        $values[] = $value;
    }

    return $values;
}

/** @return list<float> */
function cdataToFloatListWithRedundantValidation(CData $buffer, int $length): array
{
    $values = [];
    for ($index = 0; $index < $length; ++$index) {
        $value = $buffer[$index];
        if (!is_int($value) && !is_float($value)) {
            throw new RuntimeException('Native float buffer returned a non-numeric value.');
        }
        $values[] = (float) $value;
    }

    return $values;
}

/**
 * Experimental comparison only. This preserves list keys but proved slower
 * than direct CData iteration on the profiled PHP runtime.
 *
 * @return list<float>
 */
function cdataToFloatListViaUnpack(CData $buffer, int $length): array
{
    $bytes = FFI::string(FFI::cast('char *', FFI::addr($buffer[0])), $length * 4);
    /** @var array<int, float> $values */
    $values = unpack('f*', $bytes);

    return array_values($values);
}

/** @return list<float> */
function materializeTensorForComparison(FFI $ffi, CData $handle, int $length, bool $validate): array
{
    $buffer = $ffi->new("float[{$length}]");
    requireStatus(
        $ffi->transformer_tensor_copy_data_f32($handle, $buffer, $length),
        'comparison copy-out',
    );

    return $validate
        ? cdataToFloatListWithRedundantValidation($buffer, $length)
        : cdataToFloatList($buffer, $length);
}

/** @return CData Opaque TransformerTensor pointer. */
function createTensor(FFI $ffi, CData $data, CData $shape, int $rank): CData
{
    $output = $ffi->new('TransformerTensor *[1]');
    requireStatus(
        $ffi->transformer_tensor_create_f32($data, $shape, $rank, $output),
        'transformer_tensor_create_f32',
    );

    return $output[0];
}

/** @return CData Opaque TransformerTensor pointer. */
function unaryTensor(FFI $ffi, string $function, CData $input): CData
{
    $output = $ffi->new('TransformerTensor *[1]');
    requireStatus($ffi->{$function}($input, $output), $function);

    return $output[0];
}

/** @return CData Opaque TransformerTensor pointer. */
function binaryTensor(FFI $ffi, string $function, CData $a, CData $b): CData
{
    $output = $ffi->new('TransformerTensor *[1]');
    requireStatus($ffi->{$function}($a, $b, $output), $function);

    return $output[0];
}

// Shared C buffers. Their allocation and initialization are setup, not part of
// the raw-FFI timings.
$aBuffer = $ffi->new('float[' . (M * K) . ']');
$bBuffer = $ffi->new('float[' . $bLength . ']');
$residualBuffer = $ffi->new("float[{$length}]");
$matmulBuffer = $ffi->new("float[{$length}]");
$addBuffer = $ffi->new("float[{$length}]");
$transposeBuffer = $ffi->new("float[{$length}]");
$vectorBuffer = $ffi->new('float[' . N . ']');
$softmaxBuffer = $ffi->new('float[' . N . ']');
fillBuffer($aBuffer, $aValues);
fillBuffer($bBuffer, $bValues);
fillBuffer($residualBuffer, $residualValues);
fillBuffer($vectorBuffer, $vectorValues);

for ($sample = 0; $sample < $samples; ++$sample) {
    record($records, 'ffi_buffer', 'matmul', 1, static function () use (
        $ffi,
        $aBuffer,
        $bBuffer,
        $matmulBuffer,
    ): void {
        requireStatus(
            $ffi->transformer_matmul_f32($aBuffer, $bBuffer, $matmulBuffer, M, K, N),
            'transformer_matmul_f32',
        );
    });
    record($records, 'ffi_buffer', 'add', 1, static function () use (
        $ffi,
        $matmulBuffer,
        $residualBuffer,
        $addBuffer,
        $length,
    ): void {
        requireStatus(
            $ffi->transformer_tensor_add_f32($matmulBuffer, $residualBuffer, $addBuffer, $length),
            'transformer_tensor_add_f32',
        );
    });
    record($records, 'ffi_buffer', 'softmax', SOFTMAX_CALLS, static function () use (
        $ffi,
        $vectorBuffer,
        $softmaxBuffer,
    ): void {
        for ($row = 0; $row < SOFTMAX_CALLS; ++$row) {
            requireStatus(
                $ffi->transformer_softmax_f32($vectorBuffer, $softmaxBuffer, N),
                'transformer_softmax_f32',
            );
        }
    });
    record($records, 'ffi_buffer', 'transpose', 1, static function () use (
        $ffi,
        $addBuffer,
        $transposeBuffer,
    ): void {
        requireStatus(
            $ffi->transformer_transpose_f32($addBuffer, $transposeBuffer, ROWS, COLUMNS),
            'transformer_transpose_f32',
        );
    });
}

// Isolated copy-out decomposition. Setup and input creation are intentionally
// outside each measurement. Each stage reports one observable allocation where
// it creates a CData buffer or PHP array; allocator internals are not traced.
foreach ([16, 768, 98_304, 589_824, 1_048_576] as $copyOutLength) {
    $copyOutData = array_fill(0, $copyOutLength, -0.5);
    $copyOutTensor = $backend->tensorFromFloat32(
        $copyOutData,
        new \Omegaalfa\Transformer\Tensor\Shape([$copyOutLength]),
    );
    $copyOutDataBuffer = $ffi->new("float[{$copyOutLength}]");
    fillBuffer($copyOutDataBuffer, $copyOutData);
    $copyOutShape = $ffi->new('size_t[1]');
    $copyOutShape[0] = $copyOutLength;
    $copyOutHandle = createTensor($ffi, $copyOutDataBuffer, $copyOutShape, 1);
    $copyOutBuffer = $ffi->new("float[{$copyOutLength}]");
    requireStatus(
        $ffi->transformer_tensor_copy_data_f32($copyOutHandle, $copyOutBuffer, $copyOutLength),
        'copy-out setup',
    );

    $case = "n={$copyOutLength}";
    for ($sample = 0; $sample < $samples; ++$sample) {
        $exportedBuffer = null;
        record($records, "copy_out_{$case}", 'metadata_numel', 1, static function () use (
            $ffi,
            $copyOutHandle,
        ): void {
            $numel = $ffi->new('size_t[1]');
            requireStatus($ffi->transformer_tensor_numel($copyOutHandle, $numel), 'copy-out numel');
        });
        record($records, "copy_out_{$case}", 'allocate_cdata', 1, static function () use (
            $ffi,
            $copyOutLength,
        ): void {
            $ffi->new("float[{$copyOutLength}]");
        });
        record($records, "copy_out_{$case}", 'ffi_copy_preallocated', 1, static function () use (
            $ffi,
            $copyOutHandle,
            $copyOutBuffer,
            $copyOutLength,
        ): void {
            requireStatus(
                $ffi->transformer_tensor_copy_data_f32(
                    $copyOutHandle,
                    $copyOutBuffer,
                    $copyOutLength,
                ),
                'copy-out preallocated',
            );
        });
        record($records, "copy_out_{$case}", 'cdata_to_php_array_current', 1, static function () use (
            $copyOutBuffer,
            $copyOutLength,
        ): void {
            cdataToFloatListWithRedundantValidation($copyOutBuffer, $copyOutLength);
        });
        record($records, "copy_out_{$case}", 'single_guard_candidate', 1, static function () use (
            $copyOutBuffer,
            $copyOutLength,
        ): void {
            cdataToFloatList($copyOutBuffer, $copyOutLength);
        });
        record($records, "copy_out_{$case}", 'string_unpack_candidate', 1, static function () use (
            $copyOutBuffer,
            $copyOutLength,
        ): void {
            cdataToFloatListViaUnpack($copyOutBuffer, $copyOutLength);
        });
        record($records, "copy_out_{$case}", 'full_current', 1, static function () use (
            $ffi,
            $copyOutHandle,
            $copyOutLength,
        ): void {
            materializeTensorForComparison($ffi, $copyOutHandle, $copyOutLength, true);
        });
        record($records, "copy_out_{$case}", 'full_single_guard_candidate', 1, static function () use (
            $ffi,
            $copyOutHandle,
            $copyOutLength,
        ): void {
            materializeTensorForComparison($ffi, $copyOutHandle, $copyOutLength, false);
        });
        record($records, "copy_out_{$case}", 'to_float32_current', 1, static function () use (
            $copyOutTensor,
        ): void {
            $copyOutTensor->toFloat32();
        });
        record($records, "copy_out_{$case}", 'export_float32_buffer', 1, static function () use (
            $copyOutTensor,
            &$exportedBuffer,
        ): void {
            $exportedBuffer = $copyOutTensor->exportFloat32Buffer();
        });
        record($records, "copy_out_{$case}", 'buffer_value_at', 1, static function () use (
            &$exportedBuffer,
        ): void {
            $exportedBuffer->valueAt($exportedBuffer->numel() - 1);
        });
        record($records, "copy_out_{$case}", 'buffer_to_bytes', 1, static function () use (
            &$exportedBuffer,
        ): void {
            $exportedBuffer->toBytes();
        });
        record($records, "copy_out_{$case}", 'buffer_hash_consumer', 1, static function () use (
            &$exportedBuffer,
        ): void {
            hash('sha256', $exportedBuffer->toBytes());
        });
        record($records, "copy_out_{$case}", 'buffer_destroy', 1, static function () use (
            &$exportedBuffer,
        ): void {
            $exportedBuffer->destroy();
        });
    }

    if (cdataToFloatList($copyOutBuffer, $copyOutLength)
        !== cdataToFloatListViaUnpack($copyOutBuffer, $copyOutLength)) {
        throw new RuntimeException("Copy-out candidate parity failed for {$case}.");
    }
    $ffi->transformer_tensor_destroy($copyOutHandle);
    $copyOutTensor->destroy();
    unset($copyOutData, $copyOutDataBuffer, $copyOutBuffer);
}

$matrixShape = $ffi->new('size_t[2]');
$matrixShape[0] = M;
$matrixShape[1] = N;
$bShape = $ffi->new('size_t[2]');
$bShape[0] = K;
$bShape[1] = N;

// Direct, allocation-inclusive Tensor ABI comparison. Inputs are resident and
// setup is outside the timing; each result is destroyed within its measurement.
$rankOneShape = $ffi->new('size_t[1]');
$rankOneShape[0] = N;
$rankOneInput = createTensor($ffi, $vectorBuffer, $rankOneShape, 1);
$rankTwoInput = createTensor($ffi, $residualBuffer, $matrixShape, 2);
for ($sample = 0; $sample < $samples; ++$sample) {
    record($records, 'softmax_ffi_compare', 'rank1_x128', SOFTMAX_CALLS, static function () use (
        $ffi,
        $rankOneInput,
    ): void {
        for ($row = 0; $row < SOFTMAX_CALLS; ++$row) {
            $output = unaryTensor($ffi, 'transformer_tensor_softmax', $rankOneInput);
            $ffi->transformer_tensor_destroy($output);
        }
    });
    record($records, 'softmax_ffi_compare', 'rank2_x1', 1, static function () use (
        $ffi,
        $rankTwoInput,
    ): void {
        $output = unaryTensor($ffi, 'transformer_tensor_softmax_last_dim', $rankTwoInput);
        $ffi->transformer_tensor_destroy($output);
    });
}
$ffi->transformer_tensor_destroy($rankTwoInput);
$ffi->transformer_tensor_destroy($rankOneInput);

for ($sample = 0; $sample < $samples; ++$sample) {
    $inputs = [];
    record($records, 'tensor_api', 'create_inputs', 3, static function () use (
        $ffi,
        $aBuffer,
        $bBuffer,
        $residualBuffer,
        $matrixShape,
        $bShape,
        &$inputs,
    ): void {
        $inputs = [
            createTensor($ffi, $aBuffer, $matrixShape, 2),
            createTensor($ffi, $bBuffer, $bShape, 2),
            createTensor($ffi, $residualBuffer, $matrixShape, 2),
        ];
    });

    $outputs = [];
    record($records, 'tensor_api', 'matmul', 1, static function () use ($ffi, &$inputs, &$outputs): void {
        $outputs['matmul'] = binaryTensor($ffi, 'transformer_tensor_matmul', $inputs[0], $inputs[1]);
    });
    record($records, 'tensor_api', 'add', 1, static function () use ($ffi, &$inputs, &$outputs): void {
        $outputs['add'] = binaryTensor($ffi, 'transformer_tensor_add', $outputs['matmul'], $inputs[2]);
    });
    record($records, 'tensor_api', 'softmax_last_dim', 1, static function () use (
        $ffi,
        &$outputs,
    ): void {
        $outputs['softmax'] = unaryTensor(
            $ffi,
            'transformer_tensor_softmax_last_dim',
            $outputs['add'],
        );
    });
    record($records, 'tensor_api', 'transpose', 1, static function () use ($ffi, &$outputs): void {
        $outputs['transpose'] = unaryTensor($ffi, 'transformer_tensor_transpose', $outputs['add']);
    });
    record($records, 'tensor_api', 'metadata', 3, static function () use ($ffi, &$outputs): void {
        $rank = $ffi->new('size_t[1]');
        $numel = $ffi->new('size_t[1]');
        $shape = $ffi->new('size_t[2]');
        requireStatus($ffi->transformer_tensor_rank($outputs['transpose'], $rank), 'rank');
        requireStatus($ffi->transformer_tensor_numel($outputs['transpose'], $numel), 'numel');
        requireStatus($ffi->transformer_tensor_shape($outputs['transpose'], $shape, 2), 'shape');
    });
    record($records, 'tensor_api', 'copy_out', 3, static function () use (
        $ffi,
        $addBuffer,
        $transposeBuffer,
        $softmaxBuffer,
        &$outputs,
    ): void {
        requireStatus($ffi->transformer_tensor_copy_data_f32($outputs['add'], $addBuffer, M * N), 'copy add');
        requireStatus(
            $ffi->transformer_tensor_copy_data_f32($outputs['transpose'], $transposeBuffer, M * N),
            'copy transpose',
        );
        requireStatus(
            $ffi->transformer_tensor_copy_data_f32($outputs['softmax'], $addBuffer, M * N),
            'copy softmax',
        );
    });
    record($records, 'tensor_api', 'destroy_handles', 7, static function () use (
        $ffi,
        &$inputs,
        &$outputs,
    ): void {
        foreach ([$outputs['softmax'], $outputs['transpose'], $outputs['add'], $outputs['matmul'], ...$inputs] as $handle) {
            $ffi->transformer_tensor_destroy($handle);
        }
    });
}

for ($sample = 0; $sample < $samples; ++$sample) {
    $nativeInputs = [];
    record($records, 'native_php_pipeline', 'create_inputs', 3, static function () use (
        $backend,
        $aValues,
        $bValues,
        $residualValues,
        &$nativeInputs,
    ): void {
        $nativeInputs = [
            $backend->tensorFromFloat32($aValues, new \Omegaalfa\Transformer\Tensor\Shape([M, K])),
            $backend->tensorFromFloat32($bValues, new \Omegaalfa\Transformer\Tensor\Shape([K, N])),
            $backend->tensorFromFloat32($residualValues, new \Omegaalfa\Transformer\Tensor\Shape([M, N])),
        ];
    });
    $nativeOutputs = [];
    record($records, 'native_php_pipeline', 'matmul', 1, static function () use (
        $backend,
        &$nativeInputs,
        &$nativeOutputs,
    ): void {
        $nativeOutputs['matmul'] = $backend->matmul($nativeInputs[0], $nativeInputs[1]);
    });
    record($records, 'native_php_pipeline', 'add', 1, static function () use (
        $backend,
        &$nativeInputs,
        &$nativeOutputs,
    ): void {
        $nativeOutputs['add'] = $backend->add($nativeOutputs['matmul'], $nativeInputs[2]);
    });
    record($records, 'native_php_pipeline', 'softmax_last_dim', 1, static function () use (
        $backend,
        &$nativeOutputs,
    ): void {
        $nativeOutputs['softmax'] = $backend->softmax($nativeOutputs['add']);
    });
    record($records, 'native_php_pipeline', 'transpose', 1, static function () use (
        $backend,
        &$nativeOutputs,
    ): void {
        $nativeOutputs['transpose'] = $backend->transpose($nativeOutputs['add']);
    });
    record($records, 'native_php_pipeline', 'materialize_final', 1, static function () use (
        &$nativeOutputs,
    ): void {
        $nativeOutputs['materialized'] = $nativeOutputs['transpose']->toFloat32();
    });
    record($records, 'native_php_pipeline', 'export_final_buffer', 1, static function () use (
        &$nativeOutputs,
    ): void {
        $nativeOutputs['buffer'] = $nativeOutputs['transpose']->exportFloat32Buffer();
    });
    record($records, 'native_php_pipeline', 'consume_final_buffer', 1, static function () use (
        &$nativeOutputs,
    ): void {
        $nativeOutputs['buffer_hash'] = hash('sha256', $nativeOutputs['buffer']->toBytes());
    });
    record($records, 'native_php_pipeline', 'destroy_final_buffer', 1, static function () use (
        &$nativeOutputs,
    ): void {
        $nativeOutputs['buffer']->destroy();
    });
    record($records, 'native_php_pipeline', 'destroy', 7, static function () use (
        &$nativeInputs,
        &$nativeOutputs,
    ): void {
        foreach ([
            $nativeOutputs['softmax'],
            $nativeOutputs['transpose'],
            $nativeOutputs['add'],
            $nativeOutputs['matmul'],
            ...$nativeInputs,
        ] as $tensor) {
            $tensor->destroy();
        }
    });
}

for ($sample = 0; $sample < $samples; ++$sample) {
    $matmul = [];
    $added = [];
    $softmaxRows = [];
    $transposed = [];
    record($records, 'php_pipeline', 'matmul', 1, static function () use (
        $backend,
        $aValues,
        $bValues,
        &$matmul,
    ): void {
        $matmul = $backend->matmulFloat32($aValues, $bValues, M, K, N);
    });
    record($records, 'php_pipeline', 'add', 1, static function () use (
        $backend,
        $matmul,
        $residualValues,
        &$added,
    ): void {
        $added = $backend->addFloat32($matmul, $residualValues);
    });
    record($records, 'php_pipeline', 'softmax', SOFTMAX_CALLS, static function () use (
        $backend,
        $added,
        &$softmaxRows,
    ): void {
        for ($row = 0; $row < SOFTMAX_CALLS; ++$row) {
            $softmaxRows[] = $backend->softmaxFloat32(array_slice($added, $row * N, N));
        }
    });
    record($records, 'php_pipeline', 'transpose', 1, static function () use (
        $backend,
        $added,
        &$transposed,
    ): void {
        $transposed = $backend->transposeFloat32($added, ROWS, COLUMNS);
    });
}

/** @param list<float> $values */
function percentile(array $values, float $percentile): float
{
    sort($values);
    $index = (int) ceil((count($values) - 1) * $percentile);

    return $values[$index];
}

/** @var array<string, float> $layerTotals */
$layerTotals = [];
foreach ($records as $record) {
    $layerTotals[$record['layer']] = ($layerTotals[$record['layer']] ?? 0.0)
        + percentile($record['samples'], 0.5);
}

echo "profile=transformer_like M=", M, ' K=', K, ' N=', N,
    " softmax_calls=", SOFTMAX_CALLS, " samples={$samples}\n";
echo "copy_counts,php_arrays,copy_ins=133,copy_outs=131,ffi_numeric_calls=131\n";
echo "copy_counts,native_handles,copy_ins=3,copy_outs=1,ffi_numeric_calls=4\n";
echo "softmax_counts,rank1_loop,ffi_calls=128,result_allocations=128,copy_ins=0,copy_outs=0\n";
echo "softmax_counts,rank2_last_dim,ffi_calls=1,result_allocations=1,copy_ins=0,copy_outs=0\n";
echo "lifecycle,operation,vec_created,tensor_created,box_created,handle_created,handle_destroyed,allocated_bytes\n";
foreach ([
    ['matmul', 1, 1, 1, 1, 1, M * N * 4],
    ['add', 1, 1, 1, 1, 1, M * N * 4],
    ['softmax_last_dim', 1, 1, 1, 1, 1, M * N * 4],
    ['transpose', 1, 1, 1, 1, 1, M * N * 4],
    ['pipeline_outputs', 4, 4, 4, 4, 4, 4 * M * N * 4],
    ['pipeline_with_inputs', 7, 7, 7, 7, 7, (M * K + K * N + M * N + 4 * M * N) * 4],
] as [$operation, $vecs, $tensors, $boxes, $created, $destroyed, $bytes]) {
    echo "lifecycle,{$operation},{$vecs},{$tensors},{$boxes},{$created},{$destroyed},{$bytes}\n";
}
echo 'lifecycle_peaks,current_retained_handles=7,current_output_handles=4,current_output_bytes=', 4 * M * N * 4,
    ',prompt_destroy_peak_output_handles=2,prompt_destroy_peak_output_bytes=', 2 * M * N * 4, "\n";
printf(
    "pipeline_totals_us,ffi_preallocated=%.3f,native_first_pass=%.3f,native_with_array=%.3f,native_with_buffer=%.3f,native_buffer_and_hash=%.3f,native_without_export=%.3f,php_arrays=%.3f\n",
    $layerTotals['ffi_buffer'],
    $layerTotals['native_php_pipeline'],
    $layerTotals['native_php_pipeline']
        - percentile($records['native_php_pipeline:create_inputs']['samples'], 0.5)
        - percentile($records['native_php_pipeline:export_final_buffer']['samples'], 0.5)
        - percentile($records['native_php_pipeline:consume_final_buffer']['samples'], 0.5)
        - percentile($records['native_php_pipeline:destroy_final_buffer']['samples'], 0.5),
    $layerTotals['native_php_pipeline']
        - percentile($records['native_php_pipeline:create_inputs']['samples'], 0.5)
        - percentile($records['native_php_pipeline:materialize_final']['samples'], 0.5)
        - percentile($records['native_php_pipeline:consume_final_buffer']['samples'], 0.5),
    $layerTotals['native_php_pipeline']
        - percentile($records['native_php_pipeline:create_inputs']['samples'], 0.5)
        - percentile($records['native_php_pipeline:materialize_final']['samples'], 0.5),
    $layerTotals['native_php_pipeline']
        - percentile($records['native_php_pipeline:create_inputs']['samples'], 0.5)
        - percentile($records['native_php_pipeline:materialize_final']['samples'], 0.5)
        - percentile($records['native_php_pipeline:export_final_buffer']['samples'], 0.5)
        - percentile($records['native_php_pipeline:consume_final_buffer']['samples'], 0.5)
        - percentile($records['native_php_pipeline:destroy_final_buffer']['samples'], 0.5),
    $layerTotals['php_pipeline'],
);
echo "layer,stage,calls,median_us,p95_us,us_per_call,percent_of_layer\n";
foreach ($records as $key => $record) {
    [, $stage] = explode(':', $key, 2);
    $median = percentile($record['samples'], 0.5);
    $p95 = percentile($record['samples'], 0.95);
    $percent = 100.0 * $median / $layerTotals[$record['layer']];
    printf(
        "%s,%s,%d,%.3f,%.3f,%.3f,%.2f\n",
        $record['layer'],
        $stage,
        $record['calls'],
        $median,
        $p95,
        $median / $record['calls'],
        $percent,
    );
}
