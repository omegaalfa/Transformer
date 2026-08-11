<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\NN;

use LogicException;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Tensor\Tensor;

final readonly class Embedding implements Module
{
    public function __construct(public int $vocabularySize, public int $dimensions, public Runtime $runtime)
    {
    }

    public function forward(Tensor $input): Tensor
    {
        throw new LogicException('Embedding is not implemented yet.');
    }
}
