<?php

declare(strict_types=1);

use Omegaalfa\Transformer\Backend\Ffi\FfiBackend;
use Omegaalfa\Transformer\Backend\Ffi\NativeLibrary;
use Omegaalfa\Transformer\Tensor\Shape;
use Omegaalfa\Transformer\Tensor\Tensor;

require dirname(__DIR__, 2) . '/vendor/autoload.php';

const APPLICATION_M = 128;
const APPLICATION_K = 768;
const APPLICATION_N = 768;

$samples = max(3, (int) (getenv('TRANSFORMER_RESIDENT_APPLICATION_SAMPLES') ?: 25));
$minimumExecutions = max(1, (int) (getenv('TRANSFORMER_RESIDENT_APPLICATION_MIN_N') ?: 1));
$maximumExecutions = max(1, (int) (getenv('TRANSFORMER_RESIDENT_APPLICATION_MAX_N') ?: 1000));
$projectRoot = dirname(__DIR__, 2);
$backend = new FfiBackend(new NativeLibrary(NativeLibrary::defaultPath($projectRoot)));
$inputData = applicationValues(APPLICATION_M * APPLICATION_K, 0.001);
$weightData = applicationValues(APPLICATION_K * APPLICATION_N, 0.002);
$residualData = applicationValues(APPLICATION_M * APPLICATION_N, 0.003);
$inputShape = new Shape([APPLICATION_M, APPLICATION_K]);
$weightShape = new Shape([APPLICATION_K, APPLICATION_N]);
$residualShape = new Shape([APPLICATION_M, APPLICATION_N]);

/** @return list<float> */
function applicationValues(int $length, float $scale): array
{
    $values = [];
    for ($index = 0; $index < $length; ++$index) {
        $values[] = (($index % 251) - 125) * $scale;
    }

    return $values;
}

function applicationPipeline(Tensor $input, Tensor $weight, Tensor $residual): Tensor
{
    $projected = $input->matmul($weight);
    $added = $projected->add($residual);
    $normalized = $added->softmax();
    $output = $normalized->transpose();
    foreach ([$normalized, $added, $projected] as $temporary) {
        $temporary->destroy();
    }

    return $output;
}

/** @param list<Tensor> $tensors */
function applicationDestroy(array $tensors): void
{
    foreach ($tensors as $tensor) {
        $tensor->destroy();
    }
}

/** @param list<float> $values */
function applicationPercentile(array $values, int $percentile): float
{
    return $values[(int) ceil((count($values) - 1) * $percentile / 100)];
}

/**
 * @param list<float> $values
 * @return array{p50: float, p95: float, p99: float}
 */
function applicationSummary(array $values): array
{
    sort($values);

    return [
        'p50' => $values[intdiv(count($values), 2)],
        'p95' => applicationPercentile($values, 95),
        'p99' => applicationPercentile($values, 99),
    ];
}

/** @param array{p50: float, p95: float, p99: float} $summary */
function applicationReport(string $model, int $executions, string $stage, array $summary): void
{
    printf(
        "resident_application,%s,%d,%s,%.3f,%.3f,%.3f\n",
        $model,
        $executions,
        $stage,
        $summary['p50'],
        $summary['p95'],
        $summary['p99'],
    );
}

/** @return array{elapsed: float, output: Tensor} */
function applicationExecute(Tensor $input, Tensor $weight, Tensor $residual, int $executions): array
{
    $output = null;
    $started = hrtime(true);
    for ($iteration = 0; $iteration < $executions; ++$iteration) {
        $output?->destroy();
        $output = applicationPipeline($input, $weight, $residual);
    }
    if (!$output instanceof Tensor) {
        throw new LogicException('At least one resident execution is required.');
    }

    return [
        'elapsed' => (hrtime(true) - $started) / 1_000.0,
        'output' => $output,
    ];
}

echo 'resident_application_config,samples=', $samples, ',M=', APPLICATION_M,
',K=', APPLICATION_K, ',N=', APPLICATION_N, "\n";
echo "resident_application,model,executions,stage,p50_us,p95_us,p99_us\n";
echo "resident_application_counts,model,executions,input_tensors,php_to_cdata_copies,cdata_to_vec_copies,output_tensors,total_tensors\n";

foreach ([1, 10, 100, 1000] as $executions) {
    if ($executions < $minimumExecutions || $executions > $maximumExecutions) {
        continue;
    }
    $recreateTotals = [];
    $residentTotals = [];
    $residentSetups = [];
    $residentSteady = [];
    $residentTeardowns = [];
    for ($sample = 0; $sample < $samples; ++$sample) {
        $recreate = static function () use (
            $backend,
            $inputData,
            $weightData,
            $residualData,
            $inputShape,
            $weightShape,
            $residualShape,
            $executions,
        ): float {
            $started = hrtime(true);
            for ($iteration = 0; $iteration < $executions; ++$iteration) {
                $input = $backend->tensorFromFloat32($inputData, $inputShape);
                $weight = $backend->tensorFromFloat32($weightData, $weightShape);
                $residual = $backend->tensorFromFloat32($residualData, $residualShape);
                $output = applicationPipeline($input, $weight, $residual);
                applicationDestroy([$output, $residual, $weight, $input]);
            }

            return (hrtime(true) - $started) / 1_000.0;
        };
        $resident = static function () use (
            $backend,
            $inputData,
            $weightData,
            $residualData,
            $inputShape,
            $weightShape,
            $residualShape,
            $executions,
            &$residentSetups,
            &$residentSteady,
            &$residentTeardowns,
        ): float {
            $totalStarted = hrtime(true);
            $started = hrtime(true);
            $input = $backend->tensorFromFloat32($inputData, $inputShape);
            $weight = $backend->tensorFromFloat32($weightData, $weightShape);
            $residual = $backend->tensorFromFloat32($residualData, $residualShape);
            $residentSetups[] = (hrtime(true) - $started) / 1_000.0;
            $execution = applicationExecute($input, $weight, $residual, $executions);
            $residentSteady[] = $execution['elapsed'];
            $started = hrtime(true);
            applicationDestroy([$execution['output'], $residual, $weight, $input]);
            $residentTeardowns[] = (hrtime(true) - $started) / 1_000.0;

            return (hrtime(true) - $totalStarted) / 1_000.0;
        };

        if ($sample % 2 === 0) {
            $recreateTotals[] = $recreate();
            $residentTotals[] = $resident();
        } else {
            $residentTotals[] = $resident();
            $recreateTotals[] = $recreate();
        }
    }

    applicationReport('A_recreate', $executions, 'total', applicationSummary($recreateTotals));
    applicationReport('B_resident', $executions, 'total', applicationSummary($residentTotals));
    applicationReport('B_resident', $executions, 'setup', applicationSummary($residentSetups));
    applicationReport('B_resident', $executions, 'steady', applicationSummary($residentSteady));
    applicationReport('B_resident', $executions, 'teardown', applicationSummary($residentTeardowns));
    printf(
        "resident_application_counts,A_recreate,%d,%d,%d,%d,%d,%d\n",
        $executions,
        3 * $executions,
        3 * $executions,
        3 * $executions,
        4 * $executions,
        7 * $executions,
    );
    printf(
        "resident_application_counts,B_resident,%d,3,3,3,%d,%d\n",
        $executions,
        4 * $executions,
        3 + 4 * $executions,
    );
}
