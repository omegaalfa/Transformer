<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Model\Loader;

use Omegaalfa\Transformer\Model\Encoder\Bert\BertConfig;
use Omegaalfa\Transformer\Tensor\Shape;

final class BgeSmallEnV15Manifest
{
    public static function create(BertConfig $config): WeightManifest
    {
        $specs = [
            self::identity('embeddings.word_embeddings.weight', 'embeddings.word.weight', [$config->vocabularySize, $config->hiddenSize]),
            self::identity('embeddings.position_embeddings.weight', 'embeddings.position.weight', [$config->maxPositionEmbeddings, $config->hiddenSize]),
            self::identity('embeddings.token_type_embeddings.weight', 'embeddings.token_type.weight', [$config->typeVocabularySize, $config->hiddenSize]),
            self::identity('embeddings.LayerNorm.weight', 'embeddings.norm.weight', [$config->hiddenSize]),
            self::identity('embeddings.LayerNorm.bias', 'embeddings.norm.bias', [$config->hiddenSize]),
        ];
        for ($layer = 0; $layer < $config->numHiddenLayers; ++$layer) {
            $checkpoint = "encoder.layer.{$layer}.";
            $runtime = "layers.{$layer}.";
            foreach (['query', 'key', 'value'] as $projection) {
                $specs[] = self::linear($checkpoint . "attention.self.{$projection}.weight", $runtime . "attention.{$projection}.weight", $config->hiddenSize, $config->hiddenSize);
                $specs[] = self::identity($checkpoint . "attention.self.{$projection}.bias", $runtime . "attention.{$projection}.bias", [$config->hiddenSize]);
            }
            $specs[] = self::linear($checkpoint . 'attention.output.dense.weight', $runtime . 'attention.output.weight', $config->hiddenSize, $config->hiddenSize);
            $specs[] = self::identity($checkpoint . 'attention.output.dense.bias', $runtime . 'attention.output.bias', [$config->hiddenSize]);
            $specs[] = self::identity($checkpoint . 'attention.output.LayerNorm.weight', $runtime . 'attention.norm.weight', [$config->hiddenSize]);
            $specs[] = self::identity($checkpoint . 'attention.output.LayerNorm.bias', $runtime . 'attention.norm.bias', [$config->hiddenSize]);
            $specs[] = self::linear($checkpoint . 'intermediate.dense.weight', $runtime . 'ffn.input.weight', $config->hiddenSize, $config->intermediateSize);
            $specs[] = self::identity($checkpoint . 'intermediate.dense.bias', $runtime . 'ffn.input.bias', [$config->intermediateSize]);
            $specs[] = self::linear($checkpoint . 'output.dense.weight', $runtime . 'ffn.output.weight', $config->intermediateSize, $config->hiddenSize);
            $specs[] = self::identity($checkpoint . 'output.dense.bias', $runtime . 'ffn.output.bias', [$config->hiddenSize]);
            $specs[] = self::identity($checkpoint . 'output.LayerNorm.weight', $runtime . 'ffn.norm.weight', [$config->hiddenSize]);
            $specs[] = self::identity($checkpoint . 'output.LayerNorm.bias', $runtime . 'ffn.norm.bias', [$config->hiddenSize]);
        }

        return new WeightManifest($specs, ['embeddings.position_ids', 'pooler.dense.weight', 'pooler.dense.bias']);
    }

    /** @param list<int> $shape */
    private static function identity(string $checkpoint, string $runtime, array $shape): CheckpointParameterSpec
    {
        $tensorShape = new Shape($shape);
        return new CheckpointParameterSpec($checkpoint, $runtime, new WeightMaterializationSpec($tensorShape, $tensorShape, WeightOrientation::Identity));
    }

    private static function linear(string $checkpoint, string $runtime, int $input, int $output): CheckpointParameterSpec
    {
        return new CheckpointParameterSpec(
            $checkpoint,
            $runtime,
            new WeightMaterializationSpec(new Shape([$output, $input]), new Shape([$input, $output]), WeightOrientation::PyTorchLinearTranspose),
        );
    }
}
