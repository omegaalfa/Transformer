<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Model\Encoder\Bert;

use InvalidArgumentException;
use Omegaalfa\Transformer\Model\ModelConfig;
use Omegaalfa\Transformer\Model\ModelOutput;
use Omegaalfa\Transformer\NN\Module;
use Omegaalfa\Transformer\Runtime\Runtime;

final readonly class BertModel implements Module, BertModelInterface
{
    /** @param list<BertTransformerBlock> $layers */
    public function __construct(
        public BertConfig $config,
        public Runtime $runtime,
        public BertEmbeddings $embeddings,
        public array $layers,
    ) {
        if (count($layers) !== $config->numHiddenLayers || $embeddings->config !== $config || $embeddings->runtime !== $runtime) {
            throw new InvalidArgumentException('BERT model tree must match config and runtime.');
        }
        foreach ($layers as $layer) {
            if ($layer->config !== $config || $layer->runtime !== $runtime) {
                throw new InvalidArgumentException('Every BERT layer must share the model config and runtime.');
            }
        }
    }

    public function forward(BertModelInput $input): ModelOutput
    {
        $shape = $input->shape->dimensions;
        if (count($shape) !== 2) {
            throw new InvalidArgumentException('BERT model input must have shape [batch, sequence].');
        }
        if ($input->attentionMask !== null && $input->attentionMask->shape->dimensions !== $shape) {
            throw new InvalidArgumentException('BERT attention mask must match input shape.');
        }
        $hidden = $this->embeddings->forward($input->inputIds, $input->shape, $input->tokenTypeIds);
        foreach ($this->layers as $layer) {
            $hidden = $layer->forward($hidden, $input->attentionMask);
        }

        return new ModelOutput($hidden);
    }

    public function modelConfig(): ModelConfig
    {
        return $this->config->toModelConfig();
    }

    public function parameters(): array
    {
        return [];
    }

    public function modules(): array
    {
        $modules = ['embeddings' => $this->embeddings];
        foreach ($this->layers as $index => $layer) {
            $modules['layers.' . $index] = $layer;
        }

        return $modules;
    }
}
