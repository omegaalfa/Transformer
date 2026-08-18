<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Model\Encoder\Bert;

use InvalidArgumentException;
use Omegaalfa\Transformer\NN\LayerNorm;
use Omegaalfa\Transformer\NN\Module;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Tensor\Tensor;
use Omegaalfa\Transformer\Transformer\AttentionMask;

final readonly class BertTransformerBlock implements Module
{
    public function __construct(
        public BertConfig $config,
        public Runtime $runtime,
        public BertSelfAttention $attention,
        public LayerNorm $attentionNorm,
        public BertFeedForward $feedForward,
        public LayerNorm $feedForwardNorm,
    ) {
        if ($attention->config !== $config || $feedForward->config !== $config
            || $attentionNorm->normalizedSize !== $config->hiddenSize
            || $feedForwardNorm->normalizedSize !== $config->hiddenSize
            || $attention->runtime !== $runtime || $attentionNorm->runtime !== $runtime
            || $feedForward->runtime !== $runtime || $feedForwardNorm->runtime !== $runtime) {
            throw new InvalidArgumentException('BERT block components must share config, dimensions, and runtime.');
        }
    }

    public function forward(Tensor $input, ?AttentionMask $mask = null): Tensor
    {
        $attention = $this->attention->forward($input, $mask);
        $afterAttention = $this->attentionNorm->forward($input->add($attention));
        $feedForward = $this->feedForward->forward($afterAttention);

        return $this->feedForwardNorm->forward($afterAttention->add($feedForward));
    }

    public function parameters(): array
    {
        return [];
    }

    public function modules(): array
    {
        return [
            'attention' => $this->attention,
            'attention_norm' => $this->attentionNorm,
            'feed_forward' => $this->feedForward,
            'feed_forward_norm' => $this->feedForwardNorm,
        ];
    }
}
