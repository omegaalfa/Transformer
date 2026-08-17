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

$samples = max(25, (int) (getenv('TRANSFORMER_LINEAR_SAMPLES') ?: 25));
$projectRoot = dirname(__DIR__, 2);
$backend = new FfiBackend(new NativeLibrary(NativeLibrary::defaultPath($projectRoot)));
$runtime = new Runtime($backend, new RuntimeConfig(BackendType::Ffi));

/** @return list<float> */
function linearValues(int $length, float $scale): array
{
    $values = [];
    for ($index = 0; $index < $length; ++$index) {
        $values[] = (($index % 31) - 15) * $scale;
    }

    return $values;
}

/** @param list<float> $values @return array{p50: float, p95: float, p99: float} */
function linearSummary(array $values): array
{
    sort($values);

    return [
        'p50' => $values[intdiv(count($values), 2)],
        'p95' => $values[(int) ceil((count($values) - 1) * 0.95)],
        'p99' => $values[(int) ceil((count($values) - 1) * 0.99)],
    ];
}

/**
 * @param list<float> $input
 * @param list<float> $weight
 * @param list<float>|null $bias
 * @return list<float>
 */
function linearReference(array $input, array $weight, ?array $bias, int $m, int $k, int $n): array
{
    $output = array_fill(0, $m * $n, 0.0);
    for ($row = 0; $row < $m; ++$row) {
        for ($column = 0; $column < $n; ++$column) {
            $value = $bias[$column] ?? 0.0;
            for ($inner = 0; $inner < $k; ++$inner) {
                $value += $input[$row * $k + $inner] * $weight[$inner * $n + $column];
            }
            $output[$row * $n + $column] = $value;
        }
    }

    return $output;
}

echo "linear,rank,m,k,n,bias,backend,parameter_create_us,first_forward_us,steady_p50_us,steady_p95_us,steady_p99_us,gflops,export_us,php_reference_us,max_abs_error\n";
foreach ([[768, 768], [768, 3072], [3072, 768]] as [$k, $n]) {
    $weightValues = linearValues($k * $n, 0.002);
    $biasValues = linearValues($n, 0.003);
    foreach ([1, 8, 32, 128] as $m) {
        $inputValues = linearValues($m * $k, 0.001);
        $inputShape = $m === 8 ? [2, 4, $k] : [$m, $k];
        $input = $backend->tensorFromFloat32($inputValues, new Shape($inputShape));
        foreach ([false, true] as $withBias) {
            $started = hrtime(true);
            $weight = new Parameter('weight', $backend->tensorFromFloat32($weightValues, new Shape([$k, $n])));
            $bias = $withBias
                ? new Parameter('bias', $backend->tensorFromFloat32($biasValues, new Shape([$n])))
                : null;
            $linear = new Linear($k, $n, $runtime, $weight, $bias);
            $parameterCreateUs = (hrtime(true) - $started) / 1_000.0;

            $started = hrtime(true);
            $output = $linear->forward($input);
            $firstForwardUs = (hrtime(true) - $started) / 1_000.0;
            $output->destroy();
            $timings = [];
            for ($sample = 0; $sample < $samples; ++$sample) {
                $started = hrtime(true);
                $output = $linear->forward($input);
                $timings[] = (hrtime(true) - $started) / 1_000.0;
                $output->destroy();
            }
            $summary = linearSummary($timings);
            $output = $linear->forward($input);
            $started = hrtime(true);
            $actual = $output->toFloat32();
            $exportUs = (hrtime(true) - $started) / 1_000.0;
            $started = hrtime(true);
            $expected = linearReference(
                $inputValues,
                $weightValues,
                $withBias ? $biasValues : null,
                $m,
                $k,
                $n,
            );
            $referenceUs = (hrtime(true) - $started) / 1_000.0;
            $maxAbsoluteError = 0.0;
            foreach ($expected as $index => $value) {
                $maxAbsoluteError = max($maxAbsoluteError, abs($value - $actual[$index]));
            }
            printf(
                "linear,%d,%d,%d,%d,%s,dispatcher,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.9f\n",
                count($inputShape),
                $m,
                $k,
                $n,
                $withBias ? 'yes' : 'no',
                $parameterCreateUs,
                $firstForwardUs,
                $summary['p50'],
                $summary['p95'],
                $summary['p99'],
                (2.0 * $m * $k * $n) / ($summary['p50'] * 1_000.0),
                $exportUs,
                $referenceUs,
                $maxAbsoluteError,
            );
            $output->destroy();
            $bias?->tensor->destroy();
            $weight->tensor->destroy();
        }
        $input->destroy();
    }
}
