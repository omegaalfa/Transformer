<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Model;

use Omegaalfa\Transformer\Tensor\Tensor;

final readonly class ModelOutput
{
    /** @param array<string, Tensor> $auxiliary */
    public function __construct(public Tensor $lastHiddenState, public array $auxiliary = [])
    {
    }
}
