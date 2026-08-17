<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\NN;

use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Tensor\Tensor;

final readonly class Linear implements TensorModule
{
    public function __construct(
        public int $inputFeatures,
        public int $outputFeatures,
        public Runtime $runtime,
        public Parameter $weight,
        public ?Parameter $bias = null,
    ) {
        if ($inputFeatures <= 0 || $outputFeatures <= 0) {
            throw new \InvalidArgumentException('Linear dimensions must be positive.');
        }
        if ($weight->tensor->shape()->dimensions !== [$inputFeatures, $outputFeatures]) {
            throw new \InvalidArgumentException('Linear weight shape must be [input_features, output_features].');
        }
        if ($bias !== null && $bias->tensor->shape()->dimensions !== [$outputFeatures]) {
            throw new \InvalidArgumentException('Linear bias shape must be [output_features].');
        }
    }

    /** @return array<string, Parameter> */
    public function parameters(): array
    {
        $parameters = ['weight' => $this->weight];
        if ($this->bias !== null) {
            $parameters['bias'] = $this->bias;
        }

        return $parameters;
    }

    /** @return array<string, Module> */
    public function modules(): array
    {
        return [];
    }

    public function forward(Tensor $input): Tensor
    {
        return $this->runtime->backend()->linear(
            $input,
            $this->weight->tensor,
            $this->bias?->tensor,
        );
    }
}
