<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Serialization\Safetensors;

use Omegaalfa\Transformer\Tensor\DType;
use Omegaalfa\Transformer\Tensor\Shape;

final readonly class TensorMetadata
{
    /**
     * @param int $offset Absolute byte offset from the beginning of the file.
     * @param int $length Tensor payload length in bytes.
     */
    public function __construct(
        public string $name,
        public Shape $shape,
        public DType $dtype,
        public int $offset,
        public int $length,
    ) {
    }
}
