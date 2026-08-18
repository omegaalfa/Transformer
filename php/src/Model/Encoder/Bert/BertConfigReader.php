<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Model\Encoder\Bert;

use JsonException;
use Omegaalfa\Transformer\Exception\ModelException;

final readonly class BertConfigReader
{
    public function read(string $path): BertConfig
    {
        $json = @file_get_contents($path);
        if ($json === false) {
            throw new ModelException("Unable to read BERT config: {$path}");
        }
        try {
            $data = json_decode($json, true, 64, JSON_THROW_ON_ERROR);
        } catch (JsonException $exception) {
            throw new ModelException('Invalid BERT config JSON.', previous: $exception);
        }
        if (!is_array($data) || array_is_list($data)) {
            throw new ModelException('BERT config must be a JSON object.');
        }
        $data = $this->object($data);

        return new BertConfig(
            vocabularySize: $this->integer($data, 'vocab_size'),
            hiddenSize: $this->integer($data, 'hidden_size'),
            intermediateSize: $this->integer($data, 'intermediate_size'),
            numAttentionHeads: $this->integer($data, 'num_attention_heads'),
            numHiddenLayers: $this->integer($data, 'num_hidden_layers'),
            maxPositionEmbeddings: $this->integer($data, 'max_position_embeddings'),
            typeVocabularySize: $this->integer($data, 'type_vocab_size'),
            layerNormEpsilon: $this->number($data, 'layer_norm_eps'),
            activation: $this->string($data, 'hidden_act'),
            positionEmbeddingType: $this->string($data, 'position_embedding_type'),
            architecture: $this->architecture($data),
        );
    }

    /**
     * @param array<mixed, mixed> $data
     * @return array<string, mixed>
     */
    private function object(array $data): array
    {
        $object = [];
        foreach ($data as $key => $value) {
            if (!is_string($key)) {
                throw new ModelException('BERT config keys must be strings.');
            }
            $object[$key] = $value;
        }
        return $object;
    }

    /** @param array<string, mixed> $data */
    private function integer(array $data, string $key): int
    {
        $value = $data[$key] ?? null;
        if (!is_int($value)) {
            throw new ModelException("BERT config field {$key} must be an integer.");
        }
        return $value;
    }

    /** @param array<string, mixed> $data */
    private function number(array $data, string $key): float
    {
        $value = $data[$key] ?? null;
        if (!is_int($value) && !is_float($value)) {
            throw new ModelException("BERT config field {$key} must be numeric.");
        }
        return (float) $value;
    }

    /** @param array<string, mixed> $data */
    private function string(array $data, string $key): string
    {
        $value = $data[$key] ?? null;
        if (!is_string($value)) {
            throw new ModelException("BERT config field {$key} must be a string.");
        }
        return $value;
    }

    /** @param array<string, mixed> $data */
    private function architecture(array $data): string
    {
        $architectures = $data['architectures'] ?? null;
        if (!is_array($architectures) || $architectures !== ['BertModel']) {
            throw new ModelException('BERT config architectures must be exactly ["BertModel"].');
        }
        return 'BertModel';
    }
}
