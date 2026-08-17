<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Transformer;

use InvalidArgumentException;
use Omegaalfa\Transformer\NN\Linear;
use Omegaalfa\Transformer\NN\Module;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Tensor\Tensor;
use Omegaalfa\Transformer\Transformer\Config\TransformerConfig;

final readonly class MultiHeadAttention implements Module
{
    public function __construct(
        public TransformerConfig $config,
        public Runtime $runtime,
        public Linear $qProj,
        public Linear $kProj,
        public Linear $vProj,
        public Linear $outProj,
    ) {
        $dimensions = $config->hiddenSize;
        if ($dimensions <= 0 || $config->attentionHeads <= 0 || $dimensions % $config->attentionHeads !== 0) {
            throw new InvalidArgumentException('Attention requires positive hidden size and heads with hidden size divisible by heads.');
        }
        foreach ($this->projections() as $projection) {
            if ($projection->inputFeatures !== $dimensions
                || $projection->outputFeatures !== $dimensions
                || $projection->bias !== null) {
                throw new InvalidArgumentException('Attention projections must be bias-free Linear hidden_size to hidden_size.');
            }
            if ($projection->runtime !== $runtime) {
                throw new InvalidArgumentException('Attention projections must use the Attention runtime.');
            }
        }
    }

    public function forward(Tensor $input, ?AttentionMask $mask = null): Tensor
    {
        $shape = $input->shape()->dimensions;
        if (count($shape) !== 3 || $shape[2] !== $this->config->hiddenSize) {
            throw new InvalidArgumentException('Attention input must have shape [batch, sequence, hidden_size].');
        }
        if ($mask !== null && $mask->shape->dimensions !== [$shape[0], $shape[1]]) {
            throw new InvalidArgumentException('Attention mask shape must match input batch and sequence.');
        }

        return $this->runtime->backend()->multiHeadAttention(
            $input,
            $this->qProj->weight->tensor,
            $this->kProj->weight->tensor,
            $this->vProj->weight->tensor,
            $this->outProj->weight->tensor,
            $this->config->attentionHeads,
            $mask,
        );
    }

    public function parameters(): array
    {
        return [];
    }

    public function modules(): array
    {
        return [
            'q_proj' => $this->qProj,
            'k_proj' => $this->kProj,
            'v_proj' => $this->vProj,
            'out_proj' => $this->outProj,
        ];
    }

    /** @return list<Linear> */
    private function projections(): array
    {
        return [$this->qProj, $this->kProj, $this->vProj, $this->outProj];
    }
}
