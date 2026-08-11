<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\NN\Activation;

use LogicException;
use Omegaalfa\Transformer\Tensor\Tensor;

final class Gelu implements ActivationInterface
{
    public function forward(Tensor $input): Tensor
    {
        throw new LogicException('GELU is not implemented yet.');
    }
}
