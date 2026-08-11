<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tensor;

final readonly class Shape
{
    /** @param list<int> $dimensions */
    public function __construct(public array $dimensions)
    {
    }
}
