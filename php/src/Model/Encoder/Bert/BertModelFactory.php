<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Model\Encoder\Bert;

use Omegaalfa\Transformer\Exception\ModelException;
use Omegaalfa\Transformer\NN\Activation\ExactGelu;
use Omegaalfa\Transformer\NN\Embedding;
use Omegaalfa\Transformer\NN\LayerNorm;
use Omegaalfa\Transformer\NN\Linear;
use Omegaalfa\Transformer\NN\Parameter;
use Omegaalfa\Transformer\Runtime\Runtime;

final readonly class BertModelFactory
{
    public function __construct(private Runtime $runtime)
    {
    }

    /** @param array<string, Parameter> $parameters */
    public function create(BertConfig $config, array $parameters): BertModel
    {
        $embeddings = new BertEmbeddings(
            $config,
            $this->runtime,
            new Embedding($config->vocabularySize, $config->hiddenSize, $this->runtime, $this->parameter($parameters, 'embeddings.word.weight')),
            new Embedding($config->maxPositionEmbeddings, $config->hiddenSize, $this->runtime, $this->parameter($parameters, 'embeddings.position.weight')),
            new Embedding($config->typeVocabularySize, $config->hiddenSize, $this->runtime, $this->parameter($parameters, 'embeddings.token_type.weight')),
            $this->norm($config, $parameters, 'embeddings.norm'),
        );
        $layers = [];
        for ($index = 0; $index < $config->numHiddenLayers; ++$index) {
            $prefix = "layers.{$index}.";
            $attention = new BertSelfAttention(
                $config,
                $this->runtime,
                $this->linear($config->hiddenSize, $config->hiddenSize, $parameters, $prefix . 'attention.query'),
                $this->linear($config->hiddenSize, $config->hiddenSize, $parameters, $prefix . 'attention.key'),
                $this->linear($config->hiddenSize, $config->hiddenSize, $parameters, $prefix . 'attention.value'),
                $this->linear($config->hiddenSize, $config->hiddenSize, $parameters, $prefix . 'attention.output'),
            );
            $feedForward = new BertFeedForward(
                $config,
                $this->runtime,
                $this->linear($config->hiddenSize, $config->intermediateSize, $parameters, $prefix . 'ffn.input'),
                new ExactGelu($this->runtime),
                $this->linear($config->intermediateSize, $config->hiddenSize, $parameters, $prefix . 'ffn.output'),
            );
            $layers[] = new BertTransformerBlock(
                $config,
                $this->runtime,
                $attention,
                $this->norm($config, $parameters, $prefix . 'attention.norm'),
                $feedForward,
                $this->norm($config, $parameters, $prefix . 'ffn.norm'),
            );
        }

        return new BertModel($config, $this->runtime, $embeddings, $layers);
    }

    /** @param array<string, Parameter> $parameters */
    private function linear(int $input, int $output, array $parameters, string $prefix): Linear
    {
        return new Linear(
            $input,
            $output,
            $this->runtime,
            $this->parameter($parameters, $prefix . '.weight'),
            $this->parameter($parameters, $prefix . '.bias'),
        );
    }

    /** @param array<string, Parameter> $parameters */
    private function norm(BertConfig $config, array $parameters, string $prefix): LayerNorm
    {
        return new LayerNorm(
            $config->hiddenSize,
            $this->runtime,
            $this->parameter($parameters, $prefix . '.weight'),
            $this->parameter($parameters, $prefix . '.bias'),
            $config->layerNormEpsilon,
        );
    }

    /** @param array<string, Parameter> $parameters */
    private function parameter(array $parameters, string $name): Parameter
    {
        $parameter = $parameters[$name] ?? null;
        if (!$parameter instanceof Parameter || $parameter->name !== $name || $parameter->trainable) {
            throw new ModelException("Missing or invalid resident BERT Parameter: {$name}");
        }
        return $parameter;
    }
}
