<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\NN;

use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Tensor\Tensor;

final readonly class LayerNorm implements TensorModule
{
    public readonly float $epsilon;

    public function __construct(
        public int $normalizedSize,
        public Runtime $runtime,
        public Parameter $weight,
        public Parameter $bias,
        float $epsilon = 1.0e-5,
    ) {
        if ($normalizedSize <= 0) {
            throw new \InvalidArgumentException('LayerNorm normalized size must be positive.');
        }
        if ($weight->tensor->shape()->dimensions !== [$normalizedSize]
            || $bias->tensor->shape()->dimensions !== [$normalizedSize]) {
            throw new \InvalidArgumentException('LayerNorm weight and bias shapes must be [normalized_size].');
        }

        $packed = pack('g', $epsilon);
        $unpacked = unpack('gvalue', $packed);
        if ($unpacked === false || !isset($unpacked['value']) || !is_float($unpacked['value'])) {
            throw new \LogicException('Unable to validate LayerNorm epsilon as Float32.');
        }
        $rounded = $unpacked['value'];
        if (!is_finite($epsilon) || $epsilon <= 0.0 || !is_finite($rounded) || $rounded <= 0.0) {
            throw new \InvalidArgumentException('LayerNorm epsilon must be positive, finite, and representable as Float32.');
        }
        $this->epsilon = $rounded;
    }

    public function forward(Tensor $input): Tensor
    {
        $dimensions = $input->shape()->dimensions;
        if ($dimensions === [] || $dimensions[array_key_last($dimensions)] !== $this->normalizedSize) {
            throw new \InvalidArgumentException('LayerNorm input must have rank at least 1 and last dimension normalized_size.');
        }

        return $this->runtime->backend()->layerNorm(
            $input,
            $this->weight->tensor,
            $this->bias->tensor,
            $this->epsilon,
        );
    }

    public function parameters(): array
    {
        return ['weight' => $this->weight, 'bias' => $this->bias];
    }

    public function modules(): array
    {
        return [];
    }
}
