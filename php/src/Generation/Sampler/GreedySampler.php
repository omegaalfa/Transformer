<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Generation\Sampler;

use LogicException;
use Omegaalfa\Transformer\Tensor\Tensor;

final class GreedySampler implements SamplerInterface
{
    public function sample(Tensor $logits): int
    {
        throw new LogicException('Greedy sampling is a future milestone.');
    }
}
