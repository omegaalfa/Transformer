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

$samples = max(25, (int) (getenv('TRANSFORMER_LINEAR_BIAS_SAMPLES') ?: 25));
$backend = new FfiBackend(new NativeLibrary(NativeLibrary::defaultPath(dirname(__DIR__, 2))));
$runtime = new Runtime($backend, new RuntimeConfig(BackendType::Ffi));

/** @param list<float> $values @return array{p50: float, p95: float, p99: float} */
function linearBiasSummary(array $values): array
{
    sort($values);

    return [
        'p50' => $values[intdiv(count($values), 2)],
        'p95' => $values[(int) ceil((count($values) - 1) * 0.95)],
        'p99' => $values[(int) ceil((count($values) - 1) * 0.99)],
    ];
}

echo "linear_bias,m,k,n,no_bias_p50_us,no_bias_p95_us,no_bias_p99_us,bias_p50_us,bias_p95_us,bias_p99_us,median_delta_us\n";
foreach ([[768, 768], [768, 3072], [3072, 768]] as [$k, $n]) {
    $weight = new Parameter('weight', $backend->tensorFromFloat32(array_fill(0, $k * $n, 0.001), new Shape([$k, $n])));
    $bias = new Parameter('bias', $backend->tensorFromFloat32(array_fill(0, $n, 0.1), new Shape([$n])));
    $withoutBias = new Linear($k, $n, $runtime, $weight);
    $withBias = new Linear($k, $n, $runtime, $weight, $bias);
    foreach ([1, 8, 32, 128] as $m) {
        $input = $backend->tensorFromFloat32(array_fill(0, $m * $k, 0.01), new Shape([$m, $k]));
        $withoutTimes = [];
        $withTimes = [];
        for ($sample = 0; $sample < $samples; ++$sample) {
            $order = $sample % 2 === 0 ? [$withoutBias, $withBias] : [$withBias, $withoutBias];
            foreach ($order as $linear) {
                $started = hrtime(true);
                $output = $linear->forward($input);
                $elapsed = (hrtime(true) - $started) / 1_000.0;
                $output->destroy();
                if ($linear === $withoutBias) {
                    $withoutTimes[] = $elapsed;
                } else {
                    $withTimes[] = $elapsed;
                }
            }
        }
        $without = linearBiasSummary($withoutTimes);
        $with = linearBiasSummary($withTimes);
        printf(
            "linear_bias,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
            $m,
            $k,
            $n,
            $without['p50'],
            $without['p95'],
            $without['p99'],
            $with['p50'],
            $with['p95'],
            $with['p99'],
            $with['p50'] - $without['p50'],
        );
        $input->destroy();
    }
    $bias->tensor->destroy();
    $weight->tensor->destroy();
}
