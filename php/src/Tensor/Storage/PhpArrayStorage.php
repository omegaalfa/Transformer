<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tensor\Storage;

use LogicException;
use Omegaalfa\Transformer\Tensor\DType;
use Omegaalfa\Transformer\Tensor\Device;

final class PhpArrayStorage implements StorageInterface
{
    public function dtype(): DType
    {
        throw new LogicException('PHP array storage is not implemented yet.');
    }

    public function device(): Device
    {
        throw new LogicException('PHP array storage is not implemented yet.');
    }
}
