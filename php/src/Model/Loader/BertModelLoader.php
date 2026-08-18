<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Model\Loader;

use Omegaalfa\Transformer\Exception\ModelException;
use Omegaalfa\Transformer\Model\Encoder\Bert\BertConfig;
use Omegaalfa\Transformer\Model\Encoder\Bert\BertConfigReader;
use Omegaalfa\Transformer\Model\Encoder\Bert\BertModel;
use Omegaalfa\Transformer\Model\Encoder\Bert\BertModelFactory;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Serialization\Safetensors\SafetensorsReader;

final readonly class BertModelLoader
{
    public function __construct(private Runtime $runtime)
    {
    }

    public function load(string $directory): BertModel
    {
        $directory = rtrim($directory, DIRECTORY_SEPARATOR);
        $config = (new BertConfigReader())->read($directory . '/config.json');
        $this->validateTarget($config);
        $reader = new SafetensorsReader();
        $weights = (new SafetensorsWeightLoader(
            $reader,
            new WeightMaterializer($this->runtime->backend()),
            BgeSmallEnV15Manifest::create($config),
        ))->load($directory . '/model.safetensors');

        return (new BertModelFactory($this->runtime))->create($config, $weights);
    }

    private function validateTarget(BertConfig $config): void
    {
        if ($config->vocabularySize !== 30522 || $config->hiddenSize !== 384
            || $config->intermediateSize !== 1536 || $config->numAttentionHeads !== 12
            || $config->numHiddenLayers !== 12 || $config->maxPositionEmbeddings !== 512
            || $config->typeVocabularySize !== 2 || abs($config->layerNormEpsilon - 1.0e-12) > 0.0) {
            throw new ModelException('Config does not match BAAI/bge-small-en-v1.5.');
        }
    }
}
