<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\NN;

use LogicException;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Tensor\Tensor;

final readonly class RMSNorm implements TensorModule
{
    public function __construct(public int $normalizedSize, public float $epsilon, public Runtime $runtime)
    {
    }

    public function forward(Tensor $input): Tensor
    {
        throw new LogicException('RMSNorm is not implemented yet.');
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
