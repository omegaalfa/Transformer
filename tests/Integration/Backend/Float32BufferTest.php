<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tests\Integration\Backend;

use LogicException;
use Omegaalfa\Transformer\Backend\Ffi\FfiBackend;
use Omegaalfa\Transformer\Backend\Ffi\NativeLibrary;
use Omegaalfa\Transformer\Tensor\DType;
use Omegaalfa\Transformer\Tensor\Shape;
use PHPUnit\Framework\TestCase;

final class Float32BufferTest extends TestCase
{
    private FfiBackend $backend;

    protected function setUp(): void
    {
        $library = NativeLibrary::defaultPath(dirname(__DIR__, 3));
        if (!is_file($library)) {
            self::markTestSkipped('Release native runtime is not built.');
        }

        $this->backend = new FfiBackend(new NativeLibrary($library));
    }

    public function testExportsOwnedContiguousBufferWithMetadataAndContent(): void
    {
        $tensor = $this->backend->tensorFromFloat32([-1024, -1.5, 0, 1.5, 1024, 3], new Shape([2, 3]));
        $buffer = $tensor->exportFloat32Buffer();

        self::assertSame(6, $buffer->numel());
        self::assertSame(24, $buffer->sizeBytes());
        self::assertSame([2, 3], $buffer->shape()->dimensions);
        self::assertSame(DType::Float32, $buffer->dtype());
        self::assertSame(-1024.0, $buffer->valueAt(0));
        self::assertSame(3.0, $buffer->valueAt(5));
        self::assertSame([-1024.0, -1.5, 0.0, 1.5, 1024.0, 3.0], $this->decodeFloat32($buffer->toBytes()));
    }

    public function testBufferSurvivesSourceTensorDestruction(): void
    {
        $tensor = $this->backend->tensorFromFloat32([1, 2, 3], new Shape([3]));
        $buffer = $tensor->exportFloat32Buffer();

        $tensor->destroy();

        self::assertSame(2.0, $buffer->valueAt(1));
        self::assertSame([1.0, 2.0, 3.0], $this->decodeFloat32($buffer->toBytes()));
    }

    public function testDestroyingBufferDoesNotAffectTensor(): void
    {
        $tensor = $this->backend->tensorFromFloat32([1, 2, 3], new Shape([3]));
        $buffer = $tensor->exportFloat32Buffer();

        $buffer->destroy();
        $buffer->destroy();

        self::assertTrue($buffer->isDestroyed());
        self::assertSame([1.0, 2.0, 3.0], $tensor->toFloat32());
    }

    public function testRejectsBufferUseAfterDestroy(): void
    {
        $tensor = $this->backend->tensorFromFloat32([1], new Shape([1]));
        $buffer = $tensor->exportFloat32Buffer();
        $buffer->destroy();

        $this->expectException(LogicException::class);
        $buffer->toBytes();
    }

    public function testRejectsExportFromDestroyedTensor(): void
    {
        $tensor = $this->backend->tensorFromFloat32([1], new Shape([1]));
        $tensor->destroy();

        $this->expectException(LogicException::class);
        $tensor->exportFloat32Buffer();
    }

    public function testSupportsEmptyTensorAndMultipleIndependentExports(): void
    {
        $empty = $this->backend->tensorFromFloat32([], new Shape([0]));
        $emptyBuffer = $empty->exportFloat32Buffer();
        self::assertSame(0, $emptyBuffer->numel());
        self::assertSame('', $emptyBuffer->toBytes());

        $tensor = $this->backend->tensorFromFloat32([1, 2], new Shape([2]));
        $first = $tensor->exportFloat32Buffer();
        $second = $tensor->exportFloat32Buffer();
        $first->destroy();

        self::assertSame(2.0, $second->valueAt(1));
        self::assertSame($second->toBytes(), $second->toBytes());
    }

    public function testExportsRequiredLargeShapesWithoutArrayMaterialization(): void
    {
        foreach ([[128, 768], [2, 4, 768], [768, 768]] as $dimensions) {
            $length = 1;
            foreach ($dimensions as $dimension) {
                $length *= $dimension;
            }
            $tensor = $this->backend->tensorFromFloat32(array_fill(0, $length, -0.5), new Shape($dimensions));
            $buffer = $tensor->exportFloat32Buffer();

            self::assertSame($length, $buffer->numel());
            self::assertSame($dimensions, $buffer->shape()->dimensions);
            self::assertSame(-0.5, $buffer->valueAt(0));
            self::assertSame(-0.5, $buffer->valueAt($length - 1));
        }
    }

    public function testExportsChainedResultForMultipleConsumers(): void
    {
        $left = $this->backend->tensorFromFloat32([1, 2, 3, 4], new Shape([2, 2]));
        $identity = $this->backend->tensorFromFloat32([1, 0, 0, 1], new Shape([2, 2]));
        $result = $left->matmul($identity)->softmax()->transpose();
        $buffer = $result->exportFloat32Buffer();

        $firstConsumer = hash('sha256', $buffer->toBytes());
        $secondConsumer = hash('sha256', $buffer->toBytes());

        self::assertSame($firstConsumer, $secondConsumer);
        self::assertSame($result->toFloat32(), $this->decodeFloat32($buffer->toBytes()));
    }

    /** @return list<float> */
    private function decodeFloat32(string $bytes): array
    {
        $values = unpack('g*', $bytes);
        if ($values === false) {
            self::fail('Unable to decode Float32 bytes.');
        }

        $decoded = [];
        foreach ($values as $value) {
            if (!is_float($value)) {
                self::fail('Decoded Float32 value has an invalid type.');
            }
            $decoded[] = $value;
        }

        return $decoded;
    }
}
