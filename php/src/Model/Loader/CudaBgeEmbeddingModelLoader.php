<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Model\Loader;

use Omegaalfa\Transformer\Backend\Cuda\CudaBgeLibrary;
use Omegaalfa\Transformer\Embedding\CudaBgeEmbeddingModel;
use Omegaalfa\Transformer\Model\Encoder\Bert\BertConfigReader;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Serialization\Safetensors\SafetensorsReader;
use Omegaalfa\Transformer\Tokenizer\BertTokenizer;

final readonly class CudaBgeEmbeddingModelLoader
{
    public function __construct(private Runtime $cpuRuntime, private string $cudaLibraryPath)
    {
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
        $timings = [];
        $directory = rtrim($directory, DIRECTORY_SEPARATOR);
        $start = hrtime(true);
        $config = (new BertConfigReader())->read($directory . '/config.json');
        $manifest = BgeSmallEnV15Manifest::create($config);
        $timings['config_manifest'] = (hrtime(true) - $start) / 1_000.0;
        $start = hrtime(true);
        (new SafetensorsReader())->metadata($directory . '/model.safetensors');
        $timings['safetensors_header'] = (hrtime(true) - $start) / 1_000.0;
        $start = hrtime(true);
        $parameters = (new SafetensorsWeightLoader(
            new SafetensorsReader(),
            new WeightMaterializer($this->cpuRuntime->backend()),
            $manifest,
        ))->load($directory . '/model.safetensors');
        $timings['decode_materialize'] = (hrtime(true) - $start) / 1_000.0;
        $start = hrtime(true);
        $cuda = new CudaBgeLibrary($this->cudaLibraryPath);
        $timings['cuda_context_cublas'] = (hrtime(true) - $start) / 1_000.0;
        try {
            $copyoutUs = $uploadUs = 0.0;
            foreach ($manifest->parameters as $index => $spec) {
                $parameter = $parameters[$spec->parameterName];
                $start = hrtime(true);
                $values = $parameter->tensor->toFloat32();
                $copyoutUs += (hrtime(true) - $start) / 1_000.0;
                $start = hrtime(true);
                $cuda->setParameter($index, $values);
                $uploadUs += (hrtime(true) - $start) / 1_000.0;
                unset($parameters[$spec->parameterName]);
            }
            $timings['cpu_parameter_copyout'] = $copyoutUs;
            $timings['gpu_parameter_upload'] = $uploadUs;
            $start = hrtime(true);
            $cuda->finalize();
            $timings['model_finalize'] = (hrtime(true) - $start) / 1_000.0;
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
}
