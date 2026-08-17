<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Transformer;

use InvalidArgumentException;
use Omegaalfa\Transformer\NN\LayerNorm;
use Omegaalfa\Transformer\NN\Module;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Tensor\Tensor;
use Omegaalfa\Transformer\Transformer\Config\TransformerConfig;

final readonly class TransformerBlock implements Module
{
    public function __construct(
        public TransformerConfig $config,
        public Runtime $runtime,
        public LayerNorm $norm1,
        public MultiHeadAttention $attention,
        public LayerNorm $norm2,
        public FeedForward $feedForward,
    ) {
        $hiddenSize = $config->hiddenSize;
        if ($hiddenSize <= 0) {
            throw new InvalidArgumentException('TransformerBlock hidden size must be positive.');
        }
        if ($norm1->normalizedSize !== $hiddenSize || $norm2->normalizedSize !== $hiddenSize) {
            throw new InvalidArgumentException('TransformerBlock normalizations must match hidden_size.');
        }
        if ($attention->config->hiddenSize !== $hiddenSize
            || $attention->config->attentionHeads !== $config->attentionHeads
            || $config->attentionHeads <= 0
            || $hiddenSize % $config->attentionHeads !== 0) {
            throw new InvalidArgumentException('TransformerBlock Attention must match hidden_size and attention_heads.');
        }
        if ($feedForward->config->hiddenSize !== $hiddenSize
            || $feedForward->config->intermediateSize !== $config->intermediateSize
            || $feedForward->inputProjection->inputFeatures !== $hiddenSize
            || $feedForward->outputProjection->outputFeatures !== $hiddenSize) {
            throw new InvalidArgumentException('TransformerBlock FeedForward must match hidden_size and intermediate_size.');
        }
        if ($norm1->runtime !== $runtime
            || $attention->runtime !== $runtime
            || $norm2->runtime !== $runtime
            || $feedForward->runtime !== $runtime) {
            throw new InvalidArgumentException('TransformerBlock components must use the TransformerBlock runtime.');
        }
    }

    public function forward(Tensor $input, ?AttentionMask $mask = null): Tensor
    {
        $shape = $input->shape()->dimensions;
        if (count($shape) !== 3 || $shape[2] !== $this->config->hiddenSize) {
            throw new InvalidArgumentException('TransformerBlock input must have shape [batch, sequence, hidden_size].');
        }
        if ($mask !== null && $mask->shape->dimensions !== [$shape[0], $shape[1]]) {
            throw new InvalidArgumentException('TransformerBlock mask shape must match input batch and sequence.');
        }

        $normalized1 = $this->norm1->forward($input);
        $attentionOutput = $this->attention->forward($normalized1, $mask);
        $residual1 = $input->add($attentionOutput);
        $normalized2 = $this->norm2->forward($residual1);
        $feedForwardOutput = $this->feedForward->forward($normalized2);

        return $residual1->add($feedForwardOutput);
    }

    public function parameters(): array
    {
        return [];
    }

    public function modules(): array
    {
        return [
            'norm1' => $this->norm1,
            'attention' => $this->attention,
            'norm2' => $this->norm2,
            'feed_forward' => $this->feedForward,
        ];
    }
}
