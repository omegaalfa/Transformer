<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\NN;

use Omegaalfa\Transformer\Tensor\Tensor;

final readonly class Parameter
{
    public function __construct(
        public string $name,
        public Tensor $tensor,
        public bool $trainable = false,
    ) {
    }
}
