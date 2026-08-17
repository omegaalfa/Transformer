<?php

declare(strict_types=1);

use Omegaalfa\Transformer\Backend\BackendType;
use Omegaalfa\Transformer\Backend\Ffi\FfiBackend;
use Omegaalfa\Transformer\Backend\Ffi\NativeLibrary;
use Omegaalfa\Transformer\NN\Linear;
use Omegaalfa\Transformer\NN\Parameter;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Runtime\RuntimeConfig;
use Omegaalfa\Transformer\Tensor\Shape;

require dirname(__DIR__, 2) . '/vendor/autoload.php';

$samples = max(3, (int) (getenv('TRANSFORMER_LINEAR_RESIDENCY_SAMPLES') ?: 5));
$maximumN = max(1, (int) (getenv('TRANSFORMER_LINEAR_RESIDENCY_MAX_N') ?: 100));
$backend = new FfiBackend(new NativeLibrary(NativeLibrary::defaultPath(dirname(__DIR__, 2))));
$runtime = new Runtime($backend, new RuntimeConfig(BackendType::Ffi));
$inputValues = array_fill(0, 768, 0.01);
$weightValues = array_fill(0, 768 * 768, 0.001);
$biasValues = array_fill(0, 768, 0.1);
$input = $backend->tensorFromFloat32($inputValues, new Shape([1, 768]));

/** @param list<float> $values @return array{p50: float, p95: float, p99: float} */
function linearResidencySummary(array $values): array
{
    sort($values);

    return [
        'p50' => $values[intdiv(count($values), 2)],
        'p95' => $values[(int) ceil((count($values) - 1) * 0.95)],
        'p99' => $values[(int) ceil((count($values) - 1) * 0.99)],
    ];
}

echo "linear_residency,model,n,p50_us,p95_us,p99_us,weight_creations,bias_creations,output_creations\n";
foreach ([1, 10, 100] as $executions) {
    if ($executions > $maximumN) {
        continue;
    }
    $recreateTimes = [];
    $residentTimes = [];
    for ($sample = 0; $sample < $samples; ++$sample) {
        $recreate = static function () use (
            $backend,
            $runtime,
            $input,
            $weightValues,
            $biasValues,
            $executions,
        ): float {
            $started = hrtime(true);
            for ($iteration = 0; $iteration < $executions; ++$iteration) {
                $weight = new Parameter('weight', $backend->tensorFromFloat32($weightValues, new Shape([768, 768])));
                $bias = new Parameter('bias', $backend->tensorFromFloat32($biasValues, new Shape([768])));
                $linear = new Linear(768, 768, $runtime, $weight, $bias);
                $output = $linear->forward($input);
                $output->destroy();
                $bias->tensor->destroy();
                $weight->tensor->destroy();
            }

            return (hrtime(true) - $started) / 1_000.0;
        };
        $resident = static function () use (
            $backend,
            $runtime,
            $input,
            $weightValues,
            $biasValues,
            $executions,
        ): float {
            $started = hrtime(true);
            $weight = new Parameter('weight', $backend->tensorFromFloat32($weightValues, new Shape([768, 768])));
            $bias = new Parameter('bias', $backend->tensorFromFloat32($biasValues, new Shape([768])));
            $linear = new Linear(768, 768, $runtime, $weight, $bias);
            for ($iteration = 0; $iteration < $executions; ++$iteration) {
                $output = $linear->forward($input);
                $output->destroy();
            }
            $bias->tensor->destroy();
            $weight->tensor->destroy();

            return (hrtime(true) - $started) / 1_000.0;
        };
        if ($sample % 2 === 0) {
            $recreateTimes[] = $recreate();
            $residentTimes[] = $resident();
        } else {
            $residentTimes[] = $resident();
            $recreateTimes[] = $recreate();
        }
    }
    foreach ([
        ['recreate', linearResidencySummary($recreateTimes), $executions, $executions],
        ['resident', linearResidencySummary($residentTimes), 1, 1],
    ] as [$model, $summary, $weightCreations, $biasCreations]) {
        printf(
            "linear_residency,%s,%d,%.3f,%.3f,%.3f,%d,%d,%d\n",
            $model,
            $executions,
            $summary['p50'],
            $summary['p95'],
            $summary['p99'],
            $weightCreations,
            $biasCreations,
            $executions,
        );
    }
}
$input->destroy();
