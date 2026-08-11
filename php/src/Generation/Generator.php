<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Generation;

use LogicException;
use Omegaalfa\Transformer\Model\ModelInterface;
use Omegaalfa\Transformer\Tokenizer\TokenizerInterface;

final readonly class Generator
{
    public function __construct(public ModelInterface $model, public TokenizerInterface $tokenizer)
    {
    }

    public function generate(string $prompt, GenerationConfig $config): string
    {
        throw new LogicException('Autoregressive generation is a future milestone.');
    }
}
