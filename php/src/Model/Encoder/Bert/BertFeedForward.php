<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Model\Encoder\Bert;

use InvalidArgumentException;
use Omegaalfa\Transformer\NN\Activation\ExactGelu;
use Omegaalfa\Transformer\NN\Linear;
use Omegaalfa\Transformer\NN\Module;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Tensor\Tensor;

final readonly class BertFeedForward implements Module
{
    public function __construct(
        public BertConfig $config,
        public Runtime $runtime,
        public Linear $inputProjection,
        public ExactGelu $activation,
        public Linear $outputProjection,
    ) {
        if ($inputProjection->inputFeatures !== $config->hiddenSize
            || $inputProjection->outputFeatures !== $config->intermediateSize
            || $inputProjection->bias === null
            || $outputProjection->inputFeatures !== $config->intermediateSize
            || $outputProjection->outputFeatures !== $config->hiddenSize
            || $outputProjection->bias === null
            || $inputProjection->runtime !== $runtime
            || $outputProjection->runtime !== $runtime
            || $activation->runtime !== $runtime) {
            throw new InvalidArgumentException('BERT FFN requires biased D-to-I and I-to-D projections on the same runtime.');
        }
    }

    public function forward(Tensor $input): Tensor
    {
        return $this->outputProjection->forward($this->activation->forward($this->inputProjection->forward($input)));
    }

    public function parameters(): array
    {
        return [];
    }

    public function modules(): array
    {
        return ['input' => $this->inputProjection, 'activation' => $this->activation, 'output' => $this->outputProjection];
    }
}
