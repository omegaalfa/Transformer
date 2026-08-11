<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tests\Unit;

use LogicException;
use Omegaalfa\Transformer\Backend\BackendInterface;
use Omegaalfa\Transformer\Backend\PurePhp\PurePhpBackend;
use Omegaalfa\Transformer\Tensor\DType;
use Omegaalfa\Transformer\Tensor\Device;
use Omegaalfa\Transformer\Tensor\Shape;
use PHPUnit\Framework\Attributes\CoversNothing;
use PHPUnit\Framework\TestCase;

#[CoversNothing]
final class ArchitectureTest extends TestCase
{
    public function testCoreValueContractsRetainTypedConfiguration(): void
    {
        $shape = new Shape([2, 3, 768]);

        self::assertSame([2, 3, 768], $shape->dimensions);
        self::assertSame('float32', DType::Float32->value);
        self::assertSame('cuda', Device::CUDA->value);
    }

    public function testBackendSkeletonImplementsContractWithoutInventingOperations(): void
    {
        $backend = new PurePhpBackend();

        self::assertInstanceOf(BackendInterface::class, $backend);
        $this->expectException(LogicException::class);
        $backend->softmax($this->createStubTensor());
    }

    private function createStubTensor(): \Omegaalfa\Transformer\Tensor\Tensor
    {
        $storage = $this->createStub(\Omegaalfa\Transformer\Tensor\Storage\StorageInterface::class);

        return new \Omegaalfa\Transformer\Tensor\Tensor(new Shape([]), $storage);
    }
}
