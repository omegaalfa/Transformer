<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Model\Encoder\Bert;

use InvalidArgumentException;
use Omegaalfa\Transformer\Model\ModelConfig;
use Omegaalfa\Transformer\Transformer\Config\TransformerConfig;

final readonly class BertConfig
{
    public function __construct(
        public int $vocabularySize,
        public int $hiddenSize,
        public int $intermediateSize,
        public int $numAttentionHeads,
        public int $numHiddenLayers,
        public int $maxPositionEmbeddings,
        public int $typeVocabularySize,
        public float $layerNormEpsilon,
        public string $activation = 'gelu',
        public string $positionEmbeddingType = 'absolute',
        public string $architecture = 'BertModel',
    ) {
        if ($vocabularySize <= 0 || $hiddenSize <= 0 || $intermediateSize <= 0
            || $numAttentionHeads <= 0 || $numHiddenLayers <= 0
            || $maxPositionEmbeddings <= 0 || $typeVocabularySize <= 0
            || $hiddenSize % $numAttentionHeads !== 0) {
            throw new InvalidArgumentException('BERT dimensions must be positive and hidden size divisible by heads.');
        }
        if (!is_finite($layerNormEpsilon) || $layerNormEpsilon <= 0.0) {
            throw new InvalidArgumentException('BERT LayerNorm epsilon must be positive and finite.');
        }
        if ($activation !== 'gelu' || $positionEmbeddingType !== 'absolute' || $architecture !== 'BertModel') {
            throw new InvalidArgumentException('Only BertModel with exact gelu and absolute positions is supported.');
        }
    }

    public function toModelConfig(): ModelConfig
    {
        return new ModelConfig(
            architecture: $this->architecture,
            vocabularySize: $this->vocabularySize,
            transformer: new TransformerConfig(
                $this->hiddenSize,
                $this->numAttentionHeads,
                $this->intermediateSize,
                $this->numHiddenLayers,
                $this->layerNormEpsilon,
            ),
            maxPositionEmbeddings: $this->maxPositionEmbeddings,
            typeVocabularySize: $this->typeVocabularySize,
            activation: $this->activation,
            positionEmbeddingType: $this->positionEmbeddingType,
        );
    }
}
