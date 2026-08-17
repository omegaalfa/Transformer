<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\NN\Activation;

use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Tensor\Tensor;

final readonly class Gelu implements ActivationInterface
{
    public function __construct(public Runtime $runtime)
    {
    }

    public function parameters(): array
    {
        return [];
    }

    public function modules(): array
    {
        return [];
    }

    public function forward(Tensor $input): Tensor
    {
        return $this->runtime->backend()->gelu($input);
    }
}
