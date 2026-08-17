<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tests\Unit\Model;

use InvalidArgumentException;
use Omegaalfa\Transformer\Backend\AbstractBackend;
use Omegaalfa\Transformer\Backend\BackendType;
use Omegaalfa\Transformer\Exception\SerializationException;
use Omegaalfa\Transformer\Model\Loader\WeightMaterializationSpec;
use Omegaalfa\Transformer\Model\Loader\WeightMaterializer;
use Omegaalfa\Transformer\Model\Loader\WeightOrientation;
use Omegaalfa\Transformer\Serialization\Safetensors\SerializedTensor;
use Omegaalfa\Transformer\Serialization\Safetensors\TensorMetadata;
use Omegaalfa\Transformer\Tensor\DType;
use Omegaalfa\Transformer\Tensor\Device;
use Omegaalfa\Transformer\Tensor\Shape;
use Omegaalfa\Transformer\Tensor\Storage\StorageInterface;
use Omegaalfa\Transformer\Tensor\Tensor;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;

#[CoversClass(WeightMaterializer::class)]
#[CoversClass(WeightMaterializationSpec::class)]
final class WeightMaterializerTest extends TestCase
{
    public function testMaterializesIdentityFloat32WithoutChangingValues(): void
    {
        $backend = new RecordingBackend();
        $values = [0.0, -0.0, 1.5, -2.25, 1.401298464324817e-45, 3.4028234663852886e+38];
        $serialized = $this->serialized('weight', DType::Float32, [2, 3], pack('g*', ...$values));
        $spec = new WeightMaterializationSpec(new Shape([2, 3]), new Shape([2, 3]));

        $tensor = (new WeightMaterializer($backend))->materialize($serialized, $spec);

        self::assertSame([2, 3], $tensor->shape()->dimensions);
        self::assertSame($this->decode(pack('g*', ...$values)), $backend->values);
        self::assertSame([2, 3], $backend->shape?->dimensions);
        self::assertSame(pack('g', -0.0), pack('g', $backend->values[1]));
    }

    public function testTransposesPyTorchLinearWeightValuesEvenWhenStorageConversionIsRequired(): void
    {
        $backend = new RecordingBackend();
        $serialized = $this->serialized(
            'dense.weight',
            DType::Float32,
            [2, 3],
            pack('g*', 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        );
        $spec = new WeightMaterializationSpec(
            checkpointShape: new Shape([2, 3]),
            runtimeShape: new Shape([3, 2]),
            orientation: WeightOrientation::PyTorchLinearTranspose,
        );

        (new WeightMaterializer($backend))->materialize($serialized, $spec);

        self::assertSame([1.0, 4.0, 2.0, 5.0, 3.0, 6.0], $backend->values);
        self::assertSame([3, 2], $backend->shape?->dimensions);
    }

    public function testTransposesSquareWeightRatherThanTreatingEqualShapesAsIdentity(): void
    {
        $backend = new RecordingBackend();
        $serialized = $this->serialized(
            'query.weight',
            DType::Float32,
            [2, 2],
            pack('g*', 1.0, 2.0, 3.0, 4.0),
        );
        $spec = new WeightMaterializationSpec(
            new Shape([2, 2]),
            new Shape([2, 2]),
            WeightOrientation::PyTorchLinearTranspose,
        );

        (new WeightMaterializer($backend))->materialize($serialized, $spec);

        self::assertSame([1.0, 3.0, 2.0, 4.0], $backend->values);
    }

    public function testMaterializesEmptyFloat32Tensor(): void
    {
        $backend = new RecordingBackend();
        $serialized = $this->serialized('empty', DType::Float32, [0, 4], '');
        $spec = new WeightMaterializationSpec(new Shape([0, 4]), new Shape([0, 4]));

        $tensor = (new WeightMaterializer($backend))->materialize($serialized, $spec);

        self::assertSame([], $backend->values);
        self::assertSame([0, 4], $tensor->shape()->dimensions);
    }

    /** @param list<int> $shape */
    #[DataProvider('unsupportedDTypes')]
    public function testRejectsRuntimeUnsupportedDType(DType $dtype, array $shape, string $bytes): void
    {
        $backend = new RecordingBackend();
        $serialized = $this->serialized('weight', $dtype, $shape, $bytes);
        $spec = new WeightMaterializationSpec(new Shape($shape), new Shape($shape));

        try {
            (new WeightMaterializer($backend))->materialize($serialized, $spec);
            self::fail('Unsupported dtype was materialized.');
        } catch (SerializationException $exception) {
            self::assertStringContainsString('accepts only Float32', $exception->getMessage());
            self::assertSame([], $backend->values);
        }
    }

    /** @return iterable<string, array{DType, list<int>, string}> */
    public static function unsupportedDTypes(): iterable
    {
        yield 'Float16' => [DType::Float16, [1], "\0\0"];
        yield 'BFloat16' => [DType::BFloat16, [1], "\0\0"];
        yield 'Int8' => [DType::Int8, [1], "\0"];
    }

    public function testRejectsCheckpointShapeMismatchBeforePublication(): void
    {
        $backend = new RecordingBackend();
        $serialized = $this->serialized('weight', DType::Float32, [2], pack('g*', 1.0, 2.0));
        $spec = new WeightMaterializationSpec(new Shape([1, 2]), new Shape([1, 2]));

        $this->expectException(SerializationException::class);
        $this->expectExceptionMessage('checkpoint shape does not match');

        try {
            (new WeightMaterializer($backend))->materialize($serialized, $spec);
        } finally {
            self::assertSame([], $backend->values);
        }
    }

    public function testRejectsPayloadLengthMismatchBeforePublication(): void
    {
        $backend = new RecordingBackend();
        $metadata = new TensorMetadata('weight', new Shape([1]), DType::Float32, 0, 4);
        $serialized = new SerializedTensor($metadata, "\0\0");
        $spec = new WeightMaterializationSpec(new Shape([1]), new Shape([1]));

        $this->expectException(SerializationException::class);
        $this->expectExceptionMessage('payload length does not match its metadata');

        try {
            (new WeightMaterializer($backend))->materialize($serialized, $spec);
        } finally {
            self::assertSame([], $backend->values);
        }
    }

    #[DataProvider('nonFiniteValues')]
    public function testRejectsNonFiniteValuesBeforePublication(float $value): void
    {
        $backend = new RecordingBackend();
        $serialized = $this->serialized('weight', DType::Float32, [1], pack('g', $value));
        $spec = new WeightMaterializationSpec(new Shape([1]), new Shape([1]));

        $this->expectException(SerializationException::class);
        $this->expectExceptionMessage('contains a non-finite Float32 value');

        try {
            (new WeightMaterializer($backend))->materialize($serialized, $spec);
        } finally {
            self::assertSame([], $backend->values);
        }
    }

    /** @return iterable<string, array{float}> */
    public static function nonFiniteValues(): iterable
    {
        yield 'NaN' => [NAN];
        yield 'positive infinity' => [INF];
        yield 'negative infinity' => [-INF];
    }

    public function testValidInvalidValidSequenceDoesNotRetainPartialState(): void
    {
        $backend = new RecordingBackend();
        $materializer = new WeightMaterializer($backend);
        $spec = new WeightMaterializationSpec(new Shape([1]), new Shape([1]));

        $materializer->materialize($this->serialized('first', DType::Float32, [1], pack('g', 1.0)), $spec);
        self::assertSame([1.0], $backend->values);

        try {
            $materializer->materialize($this->serialized('invalid', DType::Float32, [1], pack('g', INF)), $spec);
            self::fail('Non-finite tensor was materialized.');
        } catch (SerializationException) {
            self::assertSame([1.0], $backend->values);
        }

        $materializer->materialize($this->serialized('last', DType::Float32, [1], pack('g', 2.0)), $spec);
        self::assertSame([2.0], $backend->values);
    }

    public function testSpecificationRejectsInvalidOrientationShapes(): void
    {
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Runtime shape does not match');

        new WeightMaterializationSpec(new Shape([2, 3]), new Shape([2, 3]), WeightOrientation::PyTorchLinearTranspose);
    }

    /** @param list<int> $shape */
    private function serialized(string $name, DType $dtype, array $shape, string $bytes): SerializedTensor
    {
        return new SerializedTensor(
            new TensorMetadata($name, new Shape($shape), $dtype, 0, strlen($bytes)),
            $bytes,
        );
    }

    /** @return list<float> */
    private function decode(string $bytes): array
    {
        $values = unpack('g*', $bytes);
        self::assertNotFalse($values);
        $decoded = [];

        foreach ($values as $value) {
            self::assertIsFloat($value);
            $decoded[] = $value;
        }

        return $decoded;
    }
}

final class RecordingBackend extends AbstractBackend
{
    /** @var list<float> */
    public array $values = [];
    public ?Shape $shape = null;

    public function type(): BackendType
    {
        return BackendType::PurePhp;
    }

    public function tensorFromFloat32(array $data, Shape $shape): Tensor
    {
        $this->values = $data;
        $this->shape = $shape;

        return new Tensor($shape, new RecordingStorage());
    }
}

final class RecordingStorage implements StorageInterface
{
    public function dtype(): DType
    {
        return DType::Float32;
    }

    public function device(): Device
    {
        return Device::CPU;
    }
}
