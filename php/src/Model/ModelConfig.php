<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Model;

use Omegaalfa\Transformer\Tensor\DType;
use Omegaalfa\Transformer\Tensor\Device;
use Omegaalfa\Transformer\Transformer\Config\TransformerConfig;

final readonly class ModelConfig
{
    public function __construct(
        public string $architecture,
        public int $vocabularySize,
        public TransformerConfig $transformer,
        public DType $dtype = DType::Float32,
        public Device $device = Device::CPU,
    ) {
    }
}
