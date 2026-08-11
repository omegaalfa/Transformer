<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Model;

use Omegaalfa\Transformer\Tensor\Tensor;
use Omegaalfa\Transformer\Transformer\AttentionMask;

interface ModelInterface
{
    public function config(): ModelConfig;

    public function forward(Tensor $input, ?AttentionMask $attentionMask = null): ModelOutput;
}
