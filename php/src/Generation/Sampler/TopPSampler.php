<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Generation\Sampler;

use LogicException;
use Omegaalfa\Transformer\Tensor\Tensor;

final readonly class TopPSampler implements SamplerInterface
{
    public function __construct(public float $probability)
    {
    }

    public function sample(Tensor $logits): int
    {
        throw new LogicException('Top-p sampling is a future milestone.');
    }
}
