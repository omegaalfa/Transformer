<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Model\Encoder\Bert;

use Omegaalfa\Transformer\Model\ModelConfig;
use Omegaalfa\Transformer\Model\ModelOutput;

interface BertModelInterface
{
    public function modelConfig(): ModelConfig;

    public function forward(BertModelInput $input): ModelOutput;
}
