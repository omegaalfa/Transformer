<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\NN\Activation;

use LogicException;
use Omegaalfa\Transformer\Backend\Contract\BertBackendInterface;
use Omegaalfa\Transformer\NN\TensorModule;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Tensor\Tensor;

final readonly class ExactGelu implements ActivationInterface, TensorModule
{
    public function __construct(public Runtime $runtime)
    {
    }

    public function forward(Tensor $input): Tensor
    {
        $backend = $this->runtime->backend();
        if (!$backend instanceof BertBackendInterface) {
            throw new LogicException('Selected backend does not support ExactGELU.');
        }

        return $backend->exactGelu($input);
    }

    public function parameters(): array
    {
        return [];
    }

    public function modules(): array
    {
        return [];
    }
}
