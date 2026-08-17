<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Transformer;

use InvalidArgumentException;
use Omegaalfa\Transformer\NN\Activation\Gelu;
use Omegaalfa\Transformer\NN\Linear;
use Omegaalfa\Transformer\NN\Module;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Tensor\Tensor;
use Omegaalfa\Transformer\Transformer\Config\TransformerConfig;

final readonly class FeedForward implements Module
{
    public function __construct(
        public TransformerConfig $config,
        public Runtime $runtime,
        public Linear $inputProjection,
        public Gelu $activation,
        public Linear $outputProjection,
    ) {
        $hiddenSize = $config->hiddenSize;
        $intermediateSize = $config->intermediateSize;
        if ($hiddenSize <= 0 || $intermediateSize <= 0) {
            throw new InvalidArgumentException('FeedForward hidden and intermediate sizes must be positive.');
        }
        if ($inputProjection->inputFeatures !== $hiddenSize
            || $inputProjection->outputFeatures !== $intermediateSize
            || $inputProjection->bias === null) {
            throw new InvalidArgumentException('FeedForward input projection must be Linear hidden_size to intermediate_size with bias.');
        }
        if ($outputProjection->inputFeatures !== $intermediateSize
            || $outputProjection->outputFeatures !== $hiddenSize
            || $outputProjection->bias === null) {
            throw new InvalidArgumentException('FeedForward output projection must be Linear intermediate_size to hidden_size with bias.');
        }
        if ($inputProjection->runtime !== $runtime
            || $activation->runtime !== $runtime
            || $outputProjection->runtime !== $runtime) {
            throw new InvalidArgumentException('FeedForward components must use the FeedForward runtime.');
        }
    }

    public function forward(Tensor $input): Tensor
    {
        $shape = $input->shape()->dimensions;
        if ($shape === [] || $shape[array_key_last($shape)] !== $this->config->hiddenSize) {
            throw new InvalidArgumentException('FeedForward input must have rank at least 1 and last dimension hidden_size.');
        }

        return $this->outputProjection->forward(
            $this->activation->forward(
                $this->inputProjection->forward($input),
            ),
        );
    }

    public function parameters(): array
    {
        return [];
    }

    public function modules(): array
    {
        return [
            'input_projection' => $this->inputProjection,
            'activation' => $this->activation,
            'output_projection' => $this->outputProjection,
        ];
    }
}
