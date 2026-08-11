<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Model\Auto;

use LogicException;
use Omegaalfa\Transformer\Model\ModelInterface;
use Omegaalfa\Transformer\Runtime\Runtime;

final class AutoModel
{
    public static function fromPretrained(string $path, Runtime $runtime): ModelInterface
    {
        throw new LogicException('Automatic model loading is not implemented yet.');
    }
}
