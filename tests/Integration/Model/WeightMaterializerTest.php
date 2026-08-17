<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tests\Integration\Model;

use Omegaalfa\Transformer\Backend\Ffi\FfiBackend;
use Omegaalfa\Transformer\Backend\Ffi\NativeLibrary;
use Omegaalfa\Transformer\Model\Loader\WeightMaterializationSpec;
use Omegaalfa\Transformer\Model\Loader\WeightMaterializer;
use Omegaalfa\Transformer\Model\Loader\WeightOrientation;
use Omegaalfa\Transformer\Serialization\Safetensors\SerializedTensor;
use Omegaalfa\Transformer\Serialization\Safetensors\TensorMetadata;
use Omegaalfa\Transformer\Tensor\DType;
use Omegaalfa\Transformer\Tensor\Shape;
use PHPUnit\Framework\TestCase;

final class WeightMaterializerTest extends TestCase
{
    public function testMaterializesTransposedCheckpointBytesIntoIndependentNativeTensor(): void
    {
        $library = NativeLibrary::defaultPath(dirname(__DIR__, 3));
        if (!is_file($library)) {
            self::markTestSkipped('Release native runtime is not built.');
        }

        $backend = new FfiBackend(new NativeLibrary($library));
        $bytes = pack('g*', 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        $serialized = new SerializedTensor(
            new TensorMetadata('dense.weight', new Shape([2, 3]), DType::Float32, 0, strlen($bytes)),
            $bytes,
        );
        $spec = new WeightMaterializationSpec(
            new Shape([2, 3]),
            new Shape([3, 2]),
            WeightOrientation::PyTorchLinearTranspose,
        );

        $tensor = (new WeightMaterializer($backend))->materialize($serialized, $spec);

        self::assertSame([3, 2], $tensor->shape()->dimensions);
        self::assertSame(DType::Float32, $tensor->storage()->dtype());
        self::assertSame([1.0, 4.0, 2.0, 5.0, 3.0, 6.0], $tensor->toFloat32());
        self::assertSame($bytes, $serialized->bytes);
    }
}
