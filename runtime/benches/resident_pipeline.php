<?php

declare(strict_types=1);

use FFI\CData;
use Omegaalfa\Transformer\Backend\Ffi\NativeLibrary;

require dirname(__DIR__, 2) . '/vendor/autoload.php';

const RESIDENT_M = 128;
const RESIDENT_K = 768;
const RESIDENT_N = 768;
const RESIDENT_OK = 0;

$samples = max(25, (int) (getenv('TRANSFORMER_RESIDENT_SAMPLES') ?: 25));
$counts = [1, 10, 100, 1000];
$projectRoot = dirname(__DIR__, 2);
$libraryPath = NativeLibrary::defaultPath($projectRoot);
$definitions = file_get_contents($projectRoot . '/php/src/Backend/Ffi/definitions.h');
if ($definitions === false) {
    throw new RuntimeException('Unable to read FFI definitions.');
}
$ffi = FFI::cdef($definitions, $libraryPath);
$aLength = RESIDENT_M * RESIDENT_K;
$bLength = RESIDENT_K * RESIDENT_N;
$outputLength = RESIDENT_M * RESIDENT_N;
$a = $ffi->new("float[{$aLength}]");
$b = $ffi->new("float[{$bLength}]");
$residual = $ffi->new("float[{$outputLength}]");
$matrixShape = $ffi->new('size_t[2]');
$matrixShape[0] = RESIDENT_M;
$matrixShape[1] = RESIDENT_K;
$weightShape = $ffi->new('size_t[2]');
$weightShape[0] = RESIDENT_K;
$weightShape[1] = RESIDENT_N;
$residualShape = $ffi->new('size_t[2]');
$residualShape[0] = RESIDENT_M;
$residualShape[1] = RESIDENT_N;

$fillStarted = hrtime(true);
for ($index = 0; $index < $aLength; ++$index) {
    $a[$index] = (($index % 251) - 125) * 0.001;
}
for ($index = 0; $index < $bLength; ++$index) {
    $b[$index] = (($index % 251) - 125) * 0.002;
}
for ($index = 0; $index < $outputLength; ++$index) {
    $residual[$index] = (($index % 251) - 125) * 0.003;
}
$phpToCdataUs = (hrtime(true) - $fillStarted) / 1_000.0;

function residentStatus(int $status, string $operation): void
{
    if ($status !== RESIDENT_OK) {
        throw new RuntimeException("{$operation} failed with {$status}");
    }
}

function residentCreate(FFI $ffi, CData $data, CData $shape): CData
{
    $output = $ffi->new('TransformerTensor *[1]');
    residentStatus($ffi->transformer_tensor_create_f32($data, $shape, 2, $output), 'create');

    return $output[0];
}

function residentDestroy(FFI $ffi, ?CData $handle): void
{
    if ($handle instanceof CData) {
        $ffi->transformer_tensor_destroy($handle);
    }
}

function residentPipeline(FFI $ffi, CData $input, CData $weight, CData $residual): CData
{
    $matmul = $ffi->new('TransformerTensor *[1]');
    residentStatus($ffi->transformer_tensor_matmul($input, $weight, $matmul), 'matmul');
    $added = $ffi->new('TransformerTensor *[1]');
    residentStatus($ffi->transformer_tensor_add($matmul[0], $residual, $added), 'add');
    residentDestroy($ffi, $matmul[0]);
    $softmax = $ffi->new('TransformerTensor *[1]');
    residentStatus($ffi->transformer_tensor_softmax_last_dim($added[0], $softmax), 'softmax');
    residentDestroy($ffi, $added[0]);
    $output = $ffi->new('TransformerTensor *[1]');
    residentStatus($ffi->transformer_tensor_transpose($softmax[0], $output), 'transpose');
    residentDestroy($ffi, $softmax[0]);

    return $output[0];
}

/** @return array{median: float, p95: float, p99: float, cv: float} */
function residentMeasure(int $samples, callable $operation): array
{
    $operation();
    $values = [];
    for ($sample = 0; $sample < $samples; ++$sample) {
        $started = hrtime(true);
        $operation();
        $values[] = (hrtime(true) - $started) / 1_000.0;
    }
    sort($values);
    $mean = array_sum($values) / count($values);
    $variance = array_sum(array_map(
        static fn (float $value): float => ($value - $mean) ** 2,
        $values,
    )) / count($values);

    return [
        'median' => $values[intdiv(count($values), 2)],
        'p95' => $values[(int) ceil((count($values) - 1) * 0.95)],
        'p99' => $values[(int) ceil((count($values) - 1) * 0.99)],
        'cv' => $mean === 0.0 ? 0.0 : sqrt($variance) / $mean,
    ];
}

function residentReport(string $scenario, int $executions, string $stage, array $measurement): void
{
    printf(
        "resident,%s,%d,%s,%.3f,%.3f,%.3f,%.6f,%.3f\n",
        $scenario,
        $executions,
        $stage,
        $measurement['median'],
        $measurement['p95'],
        $measurement['p99'],
        $measurement['cv'],
        $measurement['median'] / $executions,
    );
}

echo "resident_config,samples={$samples},M=", RESIDENT_M, ',K=', RESIDENT_K, ',N=', RESIDENT_N,
    ",php_to_cdata_us={$phpToCdataUs}\n";
echo "resident,scenario,executions,stage,median_us,p95_us,p99_us,cv,amortized_us\n";

$first = residentMeasure($samples, static function () use (
    $ffi,
    $a,
    $b,
    $residual,
    $matrixShape,
    $weightShape,
    $residualShape,
): void {
    $input = residentCreate($ffi, $a, $matrixShape);
    $weight = residentCreate($ffi, $b, $weightShape);
    $residualHandle = residentCreate($ffi, $residual, $residualShape);
    $output = residentPipeline($ffi, $input, $weight, $residualHandle);
    foreach ([$output, $residualHandle, $weight, $input] as $handle) {
        residentDestroy($ffi, $handle);
    }
});
residentReport('first_call', 1, 'total_no_export', $first);

function residentScenario(
    FFI $ffi,
    CData $a,
    CData $b,
    CData $residual,
    CData $matrixShape,
    CData $weightShape,
    CData $residualShape,
    int $executions,
    bool $recreateInput,
    bool $recreateWeight,
): void {
    $residentInput = $recreateInput ? null : residentCreate($ffi, $a, $matrixShape);
    $residentResidual = $recreateInput ? null : residentCreate($ffi, $residual, $residualShape);
    $residentWeight = $recreateWeight ? null : residentCreate($ffi, $b, $weightShape);
    $lastOutput = null;
    for ($iteration = 0; $iteration < $executions; ++$iteration) {
        $input = $residentInput ?? residentCreate($ffi, $a, $matrixShape);
        $residualHandle = $residentResidual ?? residentCreate($ffi, $residual, $residualShape);
        $weight = $residentWeight ?? residentCreate($ffi, $b, $weightShape);
        $output = residentPipeline($ffi, $input, $weight, $residualHandle);
        residentDestroy($ffi, $lastOutput);
        $lastOutput = $output;
        if ($recreateInput) {
            residentDestroy($ffi, $residualHandle);
            residentDestroy($ffi, $input);
        }
        if ($recreateWeight) {
            residentDestroy($ffi, $weight);
        }
    }
    $export = $ffi->new('float[' . (RESIDENT_M * RESIDENT_N) . ']');
    residentStatus(
        $ffi->transformer_tensor_copy_data_f32($lastOutput, $export, RESIDENT_M * RESIDENT_N),
        'export',
    );
    residentDestroy($ffi, $lastOutput);
    residentDestroy($ffi, $residentResidual);
    residentDestroy($ffi, $residentInput);
    residentDestroy($ffi, $residentWeight);
}

$scenarios = [
    'A_recreate_both' => [true, true],
    'B_resident_input' => [false, true],
    'C_resident_weight' => [true, false],
    'D_resident_both' => [false, false],
];
foreach ($counts as $executions) {
    $timings = array_fill_keys(array_keys($scenarios), []);
    foreach ($scenarios as [$recreateInput, $recreateWeight]) {
        residentScenario(
            $ffi, $a, $b, $residual, $matrixShape, $weightShape, $residualShape,
            $executions, $recreateInput, $recreateWeight,
        );
    }
    $scenarioNames = array_keys($scenarios);
    for ($sample = 0; $sample < $samples; ++$sample) {
        $orderedNames = array_merge(
            array_slice($scenarioNames, $sample % count($scenarioNames)),
            array_slice($scenarioNames, 0, $sample % count($scenarioNames)),
        );
        foreach ($orderedNames as $scenario) {
            [$recreateInput, $recreateWeight] = $scenarios[$scenario];
            $started = hrtime(true);
            residentScenario(
                $ffi, $a, $b, $residual, $matrixShape, $weightShape, $residualShape,
                $executions, $recreateInput, $recreateWeight,
            );
            $timings[$scenario][] = (hrtime(true) - $started) / 1_000.0;
        }
    }
    foreach ($timings as $scenario => $values) {
        sort($values);
        $mean = array_sum($values) / count($values);
        $variance = array_sum(array_map(
            static fn (float $value): float => ($value - $mean) ** 2,
            $values,
        )) / count($values);
        residentReport($scenario, $executions, 'paired_total_with_final_export', [
            'median' => $values[intdiv(count($values), 2)],
            'p95' => $values[(int) ceil((count($values) - 1) * 0.95)],
            'p99' => $values[(int) ceil((count($values) - 1) * 0.99)],
            'cv' => sqrt($variance) / $mean,
        ]);
    }
}
