<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Generation\Sampler;

use Omegaalfa\Transformer\Tensor\Tensor;

interface SamplerInterface
{
    public function sample(Tensor $logits): int;
}
