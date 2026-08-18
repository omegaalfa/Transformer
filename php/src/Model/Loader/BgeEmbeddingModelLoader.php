<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Model\Loader;

use Omegaalfa\Transformer\Embedding\BgeEmbeddingModel;
use Omegaalfa\Transformer\Embedding\BgePoolingStrategy;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Tokenizer\BertTokenizer;

final readonly class BgeEmbeddingModelLoader
{
    public function __construct(private Runtime $runtime)
    {
    }

    public function load(
        string $directory,
        BgePoolingStrategy $poolingStrategy = BgePoolingStrategy::Cls,
    ): BgeEmbeddingModel {
        $directory = rtrim($directory, DIRECTORY_SEPARATOR);
        $model = (new BertModelLoader($this->runtime))->load($directory);
        $tokenizer = BertTokenizer::fromTokenizerJson(
            $directory . '/tokenizer.json',
            $model->config->maxPositionEmbeddings,
        );

        return new BgeEmbeddingModel($model, $tokenizer, $this->runtime, $poolingStrategy);
    }
}
