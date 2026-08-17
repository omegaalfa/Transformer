<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tests\Unit\Serialization;

use Omegaalfa\Transformer\Exception\SerializationException;
use Omegaalfa\Transformer\Serialization\Safetensors\SafetensorsReader;
use Omegaalfa\Transformer\Tensor\DType;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;

#[CoversClass(SafetensorsReader::class)]
final class SafetensorsReaderTest extends TestCase
{
    /** @var list<string> */
    private array $temporaryFiles = [];

    protected function tearDown(): void
    {
        foreach ($this->temporaryFiles as $path) {
            @unlink($path);
        }
    }

    public function testReadsTensorMetadataWithoutReadingTensorValues(): void
    {
        $header = [
            '__metadata__' => ['format' => 'pt'],
            'encoder.weight' => [
                'dtype' => 'F32',
                'shape' => [2, 3],
                'data_offsets' => [0, 24],
            ],
            'encoder.bias' => [
                'dtype' => 'F16',
                'shape' => [3],
                'data_offsets' => [24, 30],
            ],
            'empty' => [
                'dtype' => 'F32',
                'shape' => [0, 3],
                'data_offsets' => [30, 30],
            ],
        ];
        $path = $this->writeSafetensors($header, str_repeat("\xA5", 30));
        $headerLength = strlen((string) json_encode($header, JSON_THROW_ON_ERROR));

        $map = (new SafetensorsReader())->metadata($path);

        self::assertSame(['encoder.weight', 'encoder.bias', 'empty'], array_keys($map->tensors));
        self::assertSame([2, 3], $map->tensors['encoder.weight']->shape->dimensions);
        self::assertSame(DType::Float32, $map->tensors['encoder.weight']->dtype);
        self::assertSame(8 + $headerLength, $map->tensors['encoder.weight']->offset);
        self::assertSame(24, $map->tensors['encoder.weight']->length);
        self::assertSame(DType::Float16, $map->tensors['encoder.bias']->dtype);
        self::assertSame(6, $map->tensors['encoder.bias']->length);
        self::assertSame(0, $map->tensors['empty']->length);
    }

    public function testAcceptsScalarTensorMetadata(): void
    {
        $path = $this->writeSafetensors([
            'scalar' => [
                'dtype' => 'F32',
                'shape' => [],
                'data_offsets' => [0, 4],
            ],
        ], "\0\0\0\0");

        $metadata = (new SafetensorsReader())->metadata($path)->tensors['scalar'];

        self::assertSame([], $metadata->shape->dimensions);
        self::assertSame(4, $metadata->length);
    }

    public function testReadsOneNamedTensorAsUnmodifiedBytes(): void
    {
        $firstBytes = "\x00\x01\x02\x03";
        $secondBytes = "\xFF\xFE";
        $path = $this->writeSafetensors([
            'first' => [
                'dtype' => 'F32',
                'shape' => [1],
                'data_offsets' => [0, 4],
            ],
            'second' => [
                'dtype' => 'F16',
                'shape' => [1],
                'data_offsets' => [4, 6],
            ],
        ], $firstBytes.$secondBytes);

        $tensor = (new SafetensorsReader())->tensor($path, 'second');

        self::assertSame('second', $tensor->metadata->name);
        self::assertSame(DType::Float16, $tensor->metadata->dtype);
        self::assertSame([1], $tensor->metadata->shape->dimensions);
        self::assertSame($secondBytes, $tensor->bytes);
    }

    public function testReadsEmptyTensorWithoutPayloadAllocation(): void
    {
        $path = $this->writeSafetensors([
            'empty' => [
                'dtype' => 'F32',
                'shape' => [0, 4],
                'data_offsets' => [0, 0],
            ],
        ], '');

        $tensor = (new SafetensorsReader())->tensor($path, 'empty');

        self::assertSame(0, $tensor->metadata->length);
        self::assertSame('', $tensor->bytes);
    }

    public function testRejectsUnknownTensorName(): void
    {
        $path = $this->writeSafetensors([
            'present' => [
                'dtype' => 'F32',
                'shape' => [1],
                'data_offsets' => [0, 4],
            ],
        ], "\0\0\0\0");

        $this->expectException(SerializationException::class);
        $this->expectExceptionMessage('tensor "missing" does not exist');

        (new SafetensorsReader())->tensor($path, 'missing');
    }

    /** @param array<string, mixed> $header */
    #[DataProvider('invalidHeaders')]
    public function testRejectsInvalidTensorMetadata(array $header, string $payload, string $message): void
    {
        $path = $this->writeSafetensors($header, $payload);

        $this->expectException(SerializationException::class);
        $this->expectExceptionMessage($message);

        (new SafetensorsReader())->metadata($path);
    }

    /** @return iterable<string, array{array<string, mixed>, string, string}> */
    public static function invalidHeaders(): iterable
    {
        yield 'unsupported dtype' => [[
            'value' => ['dtype' => 'F64', 'shape' => [1], 'data_offsets' => [0, 8]],
        ], str_repeat("\0", 8), 'unsupported Safetensors dtype'];

        yield 'negative dimension' => [[
            'value' => ['dtype' => 'F32', 'shape' => [-1], 'data_offsets' => [0, 0]],
        ], '', 'shape dimensions must be non-negative integers'];

        yield 'reversed offsets' => [[
            'value' => ['dtype' => 'F32', 'shape' => [1], 'data_offsets' => [4, 0]],
        ], str_repeat("\0", 4), 'invalid data offsets'];

        yield 'shape length mismatch' => [[
            'value' => ['dtype' => 'F32', 'shape' => [2], 'data_offsets' => [0, 4]],
        ], str_repeat("\0", 4), 'expected 8 from dtype and shape'];

        yield 'payload beyond file' => [[
            'value' => ['dtype' => 'F32', 'shape' => [2], 'data_offsets' => [0, 8]],
        ], str_repeat("\0", 4), 'extends beyond the end of the file'];

        yield 'overlapping tensors' => [[
            'first' => ['dtype' => 'F32', 'shape' => [1], 'data_offsets' => [0, 4]],
            'second' => ['dtype' => 'F32', 'shape' => [1], 'data_offsets' => [0, 4]],
        ], str_repeat("\0", 4), 'overlaps another tensor'];

        yield 'gap between tensors' => [[
            'first' => ['dtype' => 'F32', 'shape' => [1], 'data_offsets' => [0, 4]],
            'second' => ['dtype' => 'F32', 'shape' => [1], 'data_offsets' => [8, 12]],
        ], str_repeat("\0", 12), 'leaves an unindexed gap'];

        yield 'unindexed trailing bytes' => [[
            'value' => ['dtype' => 'F32', 'shape' => [1], 'data_offsets' => [0, 4]],
        ], str_repeat("\0", 8), 'unindexed trailing bytes'];
    }

    public function testRejectsTruncatedLengthPrefix(): void
    {
        $path = $this->temporaryPath();
        file_put_contents($path, "\x01\x00");

        $this->expectException(SerializationException::class);
        $this->expectExceptionMessage('Truncated Safetensors header length');

        (new SafetensorsReader())->metadata($path);
    }

    public function testRejectsHeaderThatExtendsBeyondFile(): void
    {
        $path = $this->temporaryPath();
        file_put_contents($path, pack('V2', 20, 0).'{}');

        $this->expectException(SerializationException::class);
        $this->expectExceptionMessage('header extends beyond the end of the file');

        (new SafetensorsReader())->metadata($path);
    }

    public function testRejectsMalformedJson(): void
    {
        $path = $this->temporaryPath();
        file_put_contents($path, pack('V2', 4, 0).'{bad');

        $this->expectException(SerializationException::class);
        $this->expectExceptionMessage('Invalid Safetensors header JSON');

        (new SafetensorsReader())->metadata($path);
    }

    /** @param array<string, mixed> $header */
    private function writeSafetensors(array $header, string $payload): string
    {
        $json = (string) json_encode($header, JSON_THROW_ON_ERROR);
        $path = $this->temporaryPath();
        file_put_contents($path, pack('V2', strlen($json), 0).$json.$payload);

        return $path;
    }

    private function temporaryPath(): string
    {
        $path = tempnam(sys_get_temp_dir(), 'transformer-safetensors-');

        self::assertNotFalse($path);
        $this->temporaryFiles[] = $path;

        return $path;
    }
}
