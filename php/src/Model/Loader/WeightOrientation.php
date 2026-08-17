<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Model\Loader;

enum WeightOrientation
{
    case Identity;
    case PyTorchLinearTranspose;
}
