<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tests\Integration\Backend;

use LogicException;
use Omegaalfa\Transformer\Backend\Ffi\FfiBackend;
use Omegaalfa\Transformer\Backend\Ffi\NativeLibrary;
use Omegaalfa\Transformer\Tensor\DType;
use Omegaalfa\Transformer\Tensor\Device;
use Omegaalfa\Transformer\Tensor\Shape;
use PHPUnit\Framework\TestCase;

final class NativeTensorTest extends TestCase
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

    public function testCreatesNativeTensorAndReadsMetadataWithoutMaterializing(): void
    {
        $tensor = $this->backend->tensorFromFloat32([1, 2, 3, 4, 5, 6], new Shape([2, 3]));

        self::assertSame([2, 3], $tensor->shape()->dimensions);
        self::assertSame(2, $tensor->ndim());
        self::assertSame(6, $tensor->size());
        self::assertSame(DType::Float32, $tensor->storage()->dtype());
        self::assertSame(Device::CPU, $tensor->storage()->device());
        self::assertSame([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], $tensor->toFloat32());
    }

    public function testChainsMatmulAddAndTransposeWithoutIntermediateMaterialization(): void
    {
        $a = $this->backend->tensorFromFloat32([1, 2, 3, 4, 5, 6], new Shape([2, 3]));
        $b = $this->backend->tensorFromFloat32([7, 8, 9, 10, 11, 12], new Shape([3, 2]));
        $residual = $this->backend->tensorFromFloat32([1, 1, 1, 1], new Shape([2, 2]));

        $result = $this->backend->transpose(
            $this->backend->add($this->backend->matmul($a, $b), $residual),
        );

        self::assertSame([2, 2], $result->shape()->dimensions);
        self::assertSame([59.0, 140.0, 65.0, 155.0], $result->toFloat32());
    }

    public function testNativeOperationsMatchLegacyBufferApis(): void
    {
        $library = new NativeLibrary(NativeLibrary::defaultPath(dirname(__DIR__, 3)));
        $backend = new FfiBackend($library);
        $aData = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        $bData = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        $a = $backend->tensorFromFloat32($aData, new Shape([2, 3]));
        $b = $backend->tensorFromFloat32($bData, new Shape([3, 2]));

        self::assertSame(
            $backend->matmulFloat32($aData, $bData, 2, 3, 2),
            $backend->matmul($a, $b)->toFloat32(),
        );

        $vectorData = [1.0, 2.0, 3.0];
        $vector = $backend->tensorFromFloat32($vectorData, new Shape([3]));
        $nativeSoftmax = $backend->softmax($vector)->toFloat32();
        $legacySoftmax = $backend->softmaxFloat32($vectorData);
        foreach ($nativeSoftmax as $index => $value) {
            self::assertEqualsWithDelta($legacySoftmax[$index], $value, 1.0e-6);
        }
    }

    public function testDestroyIsIdempotentAndUseAfterDestroyFails(): void
    {
        $tensor = $this->backend->tensorFromFloat32([1.0], new Shape([1]));

        $tensor->destroy();
        $tensor->destroy();

        $this->expectException(LogicException::class);
        $tensor->toFloat32();
    }

    public function testSoftmaxNormalizesRankTwoLastDimensionAndKeepsInputAlive(): void
    {
        $data = [1.0, 2.0, 3.0, -3.0, -2.0, -1.0];
        $input = $this->backend->tensorFromFloat32($data, new Shape([2, 3]));
        $output = $this->backend->softmax($input);

        self::assertSame([2, 3], $output->shape()->dimensions);
        self::assertSame(2, $output->ndim());
        $values = $output->toFloat32();
        foreach (array_chunk($values, 3) as $row) {
            self::assertEqualsWithDelta(1.0, array_sum($row), 1.0e-6);
        }
        self::assertSame($data, $input->toFloat32());
    }

    public function testRankTwoSoftmaxMatchesRankOneAppliedToEveryRow(): void
    {
        $rows = 128;
        $columns = 768;
        $data = [];
        for ($row = 0; $row < $rows; ++$row) {
            for ($column = 0; $column < $columns; ++$column) {
                $data[] = ($column % 31) * 0.01 - 0.15;
            }
        }

        $matrix = $this->backend->tensorFromFloat32($data, new Shape([$rows, $columns]));
        $actual = $matrix->softmax()->toFloat32();
        for ($row = 0; $row < $rows; ++$row) {
            $expected = $this->backend->softmaxFloat32(array_slice($data, $row * $columns, $columns));
            foreach ($expected as $column => $value) {
                self::assertEqualsWithDelta($value, $actual[$row * $columns + $column], 1.0e-6);
            }
        }
    }

    public function testSoftmaxCanChainWithoutMaterializingAndValidatesLastAxis(): void
    {
        $input = $this->backend->tensorFromFloat32([1, 2, 3, 4], new Shape([2, 2]));
        $residual = $this->backend->tensorFromFloat32([1, 1, 1, 1], new Shape([2, 2]));
        $result = $input->softmax(1)->add($residual);

        self::assertSame([2, 2], $result->shape()->dimensions);
        self::assertCount(4, $result->toFloat32());

        $this->expectException(LogicException::class);
        $input->softmax(0);
    }

    public function testRejectsInvalidShapes(): void
    {
        $left = $this->backend->tensorFromFloat32([1, 2, 3, 4], new Shape([2, 2]));
        $right = $this->backend->tensorFromFloat32([1, 2, 3], new Shape([3, 1]));

        $this->expectException(\Omegaalfa\Transformer\Exception\BackendException::class);
        $this->backend->matmul($left, $right);
    }

    public function testMaterializesRequiredShapesAndFloat32ValuesExactly(): void
    {
        foreach ([[128, 768], [2, 4, 768]] as $dimensions) {
            $length = array_product($dimensions);
            $data = [];
            $pattern = [-1024.0, -1.5, -0.0, 0.0, 1.5, 1024.0];
            for ($index = 0; $index < $length; ++$index) {
                $data[] = $pattern[$index % count($pattern)];
            }

            $tensor = $this->backend->tensorFromFloat32($data, new Shape($dimensions));

            self::assertSame($data, $tensor->toFloat32());
        }
    }

    public function testMaterializesLargeTensorMoreThanOnceWithoutConsumingIt(): void
    {
        $length = 768 * 768;
        $data = array_fill(0, $length, -0.5);
        $tensor = $this->backend->tensorFromFloat32($data, new Shape([768, 768]));

        $first = $tensor->toFloat32();
        $second = $tensor->toFloat32();

        self::assertSame($data, $first);
        self::assertSame($first, $second);
        self::assertSame([768, 768], $tensor->shape()->dimensions);
    }

    public function testMaterializesZerosAndChainedOperationResultExactly(): void
    {
        $zeros = $this->backend->tensorFromFloat32(array_fill(0, 16, 0.0), new Shape([2, 2, 4]));
        self::assertSame(array_fill(0, 16, 0.0), $zeros->toFloat32());

        $left = $this->backend->tensorFromFloat32([1, 2, 3, 4], new Shape([2, 2]));
        $right = $this->backend->tensorFromFloat32([1, 0, 0, 1], new Shape([2, 2]));
        $residual = $this->backend->tensorFromFloat32([0.5, 0.5, 0.5, 0.5], new Shape([2, 2]));
        $result = $left->matmul($right)->add($residual)->transpose();

        self::assertSame([1.5, 3.5, 2.5, 4.5], $result->toFloat32());
    }
}
