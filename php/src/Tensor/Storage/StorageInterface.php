<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tensor\Storage;

use Omegaalfa\Transformer\Tensor\DType;
use Omegaalfa\Transformer\Tensor\Device;

interface StorageInterface
{
    public function dtype(): DType;

    public function device(): Device;
}
