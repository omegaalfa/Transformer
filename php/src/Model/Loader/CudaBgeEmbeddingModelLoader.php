<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Model\Loader;

use Omegaalfa\Transformer\Backend\BackendType;
use Omegaalfa\Transformer\Backend\Cuda\CudaBgeLibrary;
use Omegaalfa\Transformer\Backend\Cuda\CudaBgePrecision;
use Omegaalfa\Transformer\Embedding\CudaBgeEmbeddingModel;
use Omegaalfa\Transformer\Exception\SerializationException;
use Omegaalfa\Transformer\Model\Encoder\Bert\BertConfigReader;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Serialization\Safetensors\SafetensorsReader;
use Omegaalfa\Transformer\Tensor\DType;
use Omegaalfa\Transformer\Tokenizer\BertTokenizer;

final readonly class CudaBgeEmbeddingModelLoader
{
    public function __construct(
        private Runtime $cpuRuntime,
        private string $cudaLibraryPath,
        private CudaBgePrecision $precision = CudaBgePrecision::Float32,
    ) {
    }

    public function load(string $directory): CudaBgeEmbeddingModel
    {
        return $this->loadInternal($directory)['model'];
    }

    /** @return array{model: CudaBgeEmbeddingModel, timings_us: array<string, float>} */
    public function benchmarkLoad(string $directory): array
    {
        return $this->loadInternal($directory);
    }

    /** @return array{model: CudaBgeEmbeddingModel, timings_us: array<string, float>} */
    private function loadInternal(string $directory): array
    {
        if ($this->cpuRuntime->config->backend !== BackendType::Ffi) {
            throw new \LogicException('CUDA BGE loading requires the native FFI runtime.');
        }
        $timings = [];
        $directory = rtrim($directory, DIRECTORY_SEPARATOR);
        $start = hrtime(true);
        $config = (new BertConfigReader())->read($directory . '/config.json');
        $manifest = BgeSmallEnV15Manifest::create($config);
        $timings['config_manifest'] = (hrtime(true) - $start) / 1_000.0;
        $checkpoint = $directory . '/model.safetensors';
        $start = hrtime(true);
        $session = (new SafetensorsReader())->open($checkpoint);
        $this->validateCheckpoint($session->weightMap->tensors, $manifest);
        $timings['safetensors_header'] = (hrtime(true) - $start) / 1_000.0;
        $start = hrtime(true);
        $cuda = new CudaBgeLibrary($this->cudaLibraryPath, $this->precision);
        $timings['cuda_context_cublas'] = (hrtime(true) - $start) / 1_000.0;
        try {
            $payloadUs = $stagingUs = 0.0;
            $payloadBytes = $peakStaging = 0;
            foreach ($manifest->parameters as $index => $spec) {
                $start = hrtime(true);
                $bytes = $session->read($spec->checkpointName);
                $payloadUs += (hrtime(true) - $start) / 1_000.0;
                $payloadBytes += strlen($bytes);
                $peakStaging = max($peakStaging, strlen($bytes));
                $shape = $spec->materialization->checkpointShape->dimensions;
                $nativeShape = count($shape) === 1 ? [$shape[0], 1] : [$shape[0], $shape[1]];
                $start = hrtime(true);
                $cuda->setParameterBytes(
                    $index,
                    $bytes,
                    $nativeShape,
                    $spec->materialization->orientation === WeightOrientation::PyTorchLinearTranspose
                );
                $stagingUs += (hrtime(true) - $start) / 1_000.0;
                unset($bytes);
            }
            $timings['payload_io'] = $payloadUs;
            $timings['payload_bytes'] = (float) $payloadBytes;
            $timings['native_staging'] = $stagingUs;
            $timings['peak_staging_bytes'] = (float) $peakStaging;
            $start = hrtime(true);
            $cuda->finalize();
            $timings['model_finalize'] = (hrtime(true) - $start) / 1_000.0;
            $diagnostics = $cuda->benchmarkDiagnostics();
            $timings['f32_validation'] = $diagnostics['parameter_validation_ns'] / 1_000.0;
            $timings['native_transpose'] = $diagnostics['parameter_transpose_ns'] / 1_000.0;
            $timings['precision_conversion'] = $diagnostics['parameter_conversion_ns'] / 1_000.0;
            $timings['gpu_parameter_upload'] = $diagnostics['parameter_upload_ns'] / 1_000.0;
            $timings['decode_materialize'] = $payloadUs + $stagingUs;
            $timings['cpu_tensor_materialization'] = 0.0;
            $start = hrtime(true);
            $tokenizer = BertTokenizer::fromTokenizerJson(
                $directory . '/tokenizer.json',
                $config->maxPositionEmbeddings,
            );
            $timings['tokenizer'] = (hrtime(true) - $start) / 1_000.0;

            return [
                'model' => new CudaBgeEmbeddingModel($cuda, $tokenizer),
                'timings_us' => $timings,
            ];
        } catch (\Throwable $exception) {
            $cuda->destroy();
            throw $exception;
        }
    }

    /** @param array<string, \Omegaalfa\Transformer\Serialization\Safetensors\TensorMetadata> $tensors */
    private function validateCheckpoint(array $tensors, WeightManifest $manifest): void
    {
        if (pack('L', 1) !== pack('V', 1)) {
            throw new SerializationException('Direct CUDA checkpoint loading requires a little-endian host.');
        }
        $expected = [];
        foreach ($manifest->parameters as $parameter) {
            $expected[$parameter->checkpointName] = true;
            $metadata = $tensors[$parameter->checkpointName] ?? null;
            if ($metadata === null || $metadata->dtype !== DType::Float32
                || $metadata->shape->dimensions !== $parameter->materialization->checkpointShape->dimensions) {
                throw new SerializationException(sprintf('Tensor "%s" is missing or incompatible.', $parameter->checkpointName));
            }
        }
        foreach ($manifest->ignoredCheckpointTensors as $name) {
            $expected[$name] = true;
        }
        $missing = array_keys(array_diff_key($expected, $tensors));
        $unexpected = array_keys(array_diff_key($tensors, $expected));
        if ($missing !== [] || $unexpected !== []) {
            throw new SerializationException(sprintf(
                'CUDA checkpoint manifest mismatch; missing=[%s], unexpected=[%s].',
                implode(', ', $missing),
                implode(', ', $unexpected)
            ));
        }
    }
}
