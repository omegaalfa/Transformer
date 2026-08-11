<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Transformer;

use LogicException;
use Omegaalfa\Transformer\NN\Activation\ActivationInterface;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Tensor\Tensor;
use Omegaalfa\Transformer\Transformer\Config\TransformerConfig;

final readonly class FeedForward
{
    public function __construct(
        public TransformerConfig $config,
        public ActivationInterface $activation,
        public Runtime $runtime,
    ) {
    }

    public function forward(Tensor $input): Tensor
    {
        throw new LogicException('Feed-forward is not implemented yet.');
    }
}
