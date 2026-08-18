<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Model\Encoder\Bert;

use InvalidArgumentException;
use LogicException;
use Omegaalfa\Transformer\Backend\Contract\BertBackendInterface;
use Omegaalfa\Transformer\NN\Linear;
use Omegaalfa\Transformer\NN\Module;
use Omegaalfa\Transformer\NN\Parameter;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Tensor\Tensor;
use Omegaalfa\Transformer\Transformer\AttentionMask;

final readonly class BertSelfAttention implements Module
{
    public function __construct(
        public BertConfig $config,
        public Runtime $runtime,
        public Linear $query,
        public Linear $key,
        public Linear $value,
        public Linear $output,
    ) {
        foreach ($this->projections() as $projection) {
            if ($projection->inputFeatures !== $config->hiddenSize
                || $projection->outputFeatures !== $config->hiddenSize
                || $projection->bias === null
                || $projection->runtime !== $runtime) {
                throw new InvalidArgumentException('BERT Attention projections must be biased hidden_size Linear modules on the same runtime.');
            }
        }
    }

    public function forward(Tensor $input, ?AttentionMask $mask = null): Tensor
    {
        $shape = $input->shape()->dimensions;
        if (count($shape) !== 3 || $shape[2] !== $this->config->hiddenSize) {
            throw new InvalidArgumentException('BERT Attention input must have shape [batch, sequence, hidden_size].');
        }
        if ($mask !== null && $mask->shape->dimensions !== [$shape[0], $shape[1]]) {
            throw new InvalidArgumentException('BERT Attention mask must match batch and sequence.');
        }
        $backend = $this->runtime->backend();
        if (!$backend instanceof BertBackendInterface) {
            throw new LogicException('Selected backend does not support BERT Attention.');
        }

        return $backend->bertSelfAttention(
            $input,
            $this->query->weight->tensor,
            $this->bias($this->query)->tensor,
            $this->key->weight->tensor,
            $this->bias($this->key)->tensor,
            $this->value->weight->tensor,
            $this->bias($this->value)->tensor,
            $this->output->weight->tensor,
            $this->bias($this->output)->tensor,
            $this->config->numAttentionHeads,
            $mask,
        );
    }

    public function parameters(): array
    {
        return [];
    }

    public function modules(): array
    {
        return ['query' => $this->query, 'key' => $this->key, 'value' => $this->value, 'output' => $this->output];
    }

    /** @return list<Linear> */
    private function projections(): array
    {
        return [$this->query, $this->key, $this->value, $this->output];
    }

    private function bias(Linear $projection): Parameter
    {
        return $projection->bias ?? throw new LogicException('Validated BERT projection lost its bias.');
    }
}
