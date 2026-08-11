<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Generation\Sampler;

use LogicException;
use Omegaalfa\Transformer\Tensor\Tensor;

final readonly class TopKSampler implements SamplerInterface
{
    public function __construct(public int $k)
    {
    }

    public function sample(Tensor $logits): int
    {
        throw new LogicException('Top-k sampling is a future milestone.');
    }
}
