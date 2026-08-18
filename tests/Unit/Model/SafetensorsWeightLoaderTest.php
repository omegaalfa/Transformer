<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tests\Unit\Model;

use InvalidArgumentException;
use Omegaalfa\Transformer\Backend\AbstractBackend;
use Omegaalfa\Transformer\Backend\BackendType;
use Omegaalfa\Transformer\Exception\SerializationException;
use Omegaalfa\Transformer\Model\Loader\CheckpointParameterSpec;
use Omegaalfa\Transformer\Model\Loader\SafetensorsWeightLoader;
use Omegaalfa\Transformer\Model\Loader\WeightManifest;
use Omegaalfa\Transformer\Model\Loader\WeightMaterializationSpec;
use Omegaalfa\Transformer\Model\Loader\WeightMaterializer;
use Omegaalfa\Transformer\Model\Loader\WeightOrientation;
use Omegaalfa\Transformer\Serialization\Safetensors\SafetensorsReader;
use Omegaalfa\Transformer\Tensor\DType;
use Omegaalfa\Transformer\Tensor\Device;
use Omegaalfa\Transformer\Tensor\Shape;
use Omegaalfa\Transformer\Tensor\Storage\StorageInterface;
use Omegaalfa\Transformer\Tensor\Tensor;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\TestCase;

#[CoversClass(SafetensorsWeightLoader::class)]
#[CoversClass(WeightManifest::class)]
#[CoversClass(CheckpointParameterSpec::class)]
final class SafetensorsWeightLoaderTest extends TestCase
{
    /** @var list<string> */
    private array $temporaryFiles = [];

    protected function tearDown(): void
    {
        foreach ($this->temporaryFiles as $path) {
            @unlink($path);
        }
    }

    public function testLoadsClosedManifestIntoNamedResidentParameters(): void
    {
        $path = $this->writeSafetensors([
            'encoder.bias' => [
                'dtype' => 'F32',
                'shape' => [2],
                'data_offsets' => [0, 8],
            ],
            'encoder.weight' => [
                'dtype' => 'F32',
                'shape' => [2, 3],
                'data_offsets' => [8, 32],
            ],
        ], pack('g*', 0.5, -0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0));
        $backend = new LoaderRecordingBackend();
        $loader = $this->loader($backend, new WeightManifest([
            new CheckpointParameterSpec(
                'encoder.weight',
                'encoder.weight',
                new WeightMaterializationSpec(
                    new Shape([2, 3]),
                    new Shape([3, 2]),
                    WeightOrientation::PyTorchLinearTranspose,
                ),
            ),
            new CheckpointParameterSpec(
                'encoder.bias',
                'encoder.bias',
                new WeightMaterializationSpec(new Shape([2]), new Shape([2])),
            ),
        ]));

        $parameters = $loader->load($path);

        self::assertSame(['encoder.weight', 'encoder.bias'], array_keys($parameters));
        self::assertSame('encoder.weight', $parameters['encoder.weight']->name);
        self::assertFalse($parameters['encoder.weight']->trainable);
        self::assertSame([3, 2], $parameters['encoder.weight']->tensor->shape()->dimensions);
        self::assertSame([2], $parameters['encoder.bias']->tensor->shape()->dimensions);
        self::assertSame([
            [1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
            [0.5, -0.5],
        ], $backend->materializedValues);
    }

    public function testRejectsMissingTensorBeforeMaterializingAnything(): void
    {
        $path = $this->writeSafetensors([
            'present' => ['dtype' => 'F32', 'shape' => [1], 'data_offsets' => [0, 4]],
        ], pack('g', 1.0));
        $backend = new LoaderRecordingBackend();
        $loader = $this->loader($backend, new WeightManifest([
            $this->identitySpec('present', 'present', [1]),
            $this->identitySpec('missing', 'missing', [1]),
        ]));

        $this->expectException(SerializationException::class);
        $this->expectExceptionMessage('missing required tensors: missing');

        try {
            $loader->load($path);
        } finally {
            self::assertSame([], $backend->materializedValues);
        }
    }

    public function testRejectsUnexpectedTensorBeforeMaterializingAnything(): void
    {
        $path = $this->writeSafetensors([
            'expected' => ['dtype' => 'F32', 'shape' => [1], 'data_offsets' => [0, 4]],
            'extra' => ['dtype' => 'F32', 'shape' => [1], 'data_offsets' => [4, 8]],
        ], pack('g*', 1.0, 2.0));
        $backend = new LoaderRecordingBackend();
        $loader = $this->loader($backend, new WeightManifest([
            $this->identitySpec('expected', 'expected', [1]),
        ]));

        $this->expectException(SerializationException::class);
        $this->expectExceptionMessage('unexpected tensors: extra');

        try {
            $loader->load($path);
        } finally {
            self::assertSame([], $backend->materializedValues);
        }
    }

    public function testAllowsOnlyExplicitlyIgnoredNonParameterTensors(): void
    {
        $path = $this->writeSafetensors([
            'weight' => ['dtype' => 'F32', 'shape' => [1], 'data_offsets' => [0, 4]],
            'position_ids' => ['dtype' => 'I64', 'shape' => [1], 'data_offsets' => [4, 12]],
        ], pack('g', 2.0).str_repeat("\0", 8));
        $backend = new LoaderRecordingBackend();
        $loader = $this->loader($backend, new WeightManifest([
            $this->identitySpec('weight', 'weight', [1]),
        ], ['position_ids']));

        $parameters = $loader->load($path);

        self::assertSame(['weight'], array_keys($parameters));
        self::assertSame([[2.0]], $backend->materializedValues);
    }

    public function testRejectsShapeMismatchBeforeMaterializingAnything(): void
    {
        $path = $this->writeSafetensors([
            'weight' => ['dtype' => 'F32', 'shape' => [2], 'data_offsets' => [0, 8]],
        ], pack('g*', 1.0, 2.0));
        $backend = new LoaderRecordingBackend();
        $loader = $this->loader($backend, new WeightManifest([
            $this->identitySpec('weight', 'weight', [1, 2]),
        ]));

        $this->expectException(SerializationException::class);
        $this->expectExceptionMessage('shape does not match the weight manifest');

        try {
            $loader->load($path);
        } finally {
            self::assertSame([], $backend->materializedValues);
        }
    }

    public function testManifestRejectsDuplicateCheckpointNames(): void
    {
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Duplicate checkpoint tensor name');

        new WeightManifest([
            $this->identitySpec('same', 'first', [1]),
            $this->identitySpec('same', 'second', [1]),
        ]);
    }

    public function testManifestRejectsDuplicateParameterNames(): void
    {
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Duplicate Parameter name');

        new WeightManifest([
            $this->identitySpec('first', 'same', [1]),
            $this->identitySpec('second', 'same', [1]),
        ]);
    }

    public function testValidInvalidValidLoadHasNoPublishedInvalidMap(): void
    {
        $validPath = $this->writeSafetensors([
            'weight' => ['dtype' => 'F32', 'shape' => [1], 'data_offsets' => [0, 4]],
        ], pack('g', 1.0));
        $invalidPath = $this->writeSafetensors([
            'weight' => ['dtype' => 'F32', 'shape' => [1], 'data_offsets' => [0, 4]],
        ], pack('g', INF));
        $backend = new LoaderRecordingBackend();
        $loader = $this->loader($backend, new WeightManifest([
            $this->identitySpec('weight', 'weight', [1]),
        ]));

        self::assertSame(['weight'], array_keys($loader->load($validPath)));

        try {
            $loader->load($invalidPath);
            self::fail('Invalid checkpoint was published.');
        } catch (SerializationException) {
            self::assertCount(1, $backend->materializedValues);
        }

        self::assertSame(['weight'], array_keys($loader->load($validPath)));
        self::assertCount(2, $backend->materializedValues);
    }

    private function loader(LoaderRecordingBackend $backend, WeightManifest $manifest): SafetensorsWeightLoader
    {
        return new SafetensorsWeightLoader(
            new SafetensorsReader(),
            new WeightMaterializer($backend),
            $manifest,
        );
    }

    /** @param list<int> $shape */
    private function identitySpec(string $checkpointName, string $parameterName, array $shape): CheckpointParameterSpec
    {
        return new CheckpointParameterSpec(
            $checkpointName,
            $parameterName,
            new WeightMaterializationSpec(new Shape($shape), new Shape($shape)),
        );
    }

    /** @param array<string, mixed> $header */
    private function writeSafetensors(array $header, string $payload): string
    {
        $json = (string) json_encode($header, JSON_THROW_ON_ERROR);
        $path = tempnam(sys_get_temp_dir(), 'transformer-weight-loader-');
        self::assertNotFalse($path);
        $this->temporaryFiles[] = $path;
        file_put_contents($path, pack('V2', strlen($json), 0).$json.$payload);

        return $path;
    }
}

final class LoaderRecordingBackend extends AbstractBackend
{
    /** @var list<list<float>> */
    public array $materializedValues = [];

    public function type(): BackendType
    {
        return BackendType::PurePhp;
    }

    public function tensorFromFloat32(array $data, Shape $shape): Tensor
    {
        $this->materializedValues[] = $data;

        return new Tensor($shape, new LoaderRecordingStorage());
    }
}

final class LoaderRecordingStorage implements StorageInterface
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
