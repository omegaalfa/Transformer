<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\NN;

use LogicException;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Tensor\Tensor;

final readonly class Linear implements Module
{
    public function __construct(
        public int $inputFeatures,
        public int $outputFeatures,
        public bool $bias,
        public Runtime $runtime,
    ) {
    }

    public function forward(Tensor $input): Tensor
    {
        throw new LogicException('Linear is not implemented yet.');
    }
}
