<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Backend\Ffi;

use FFI;
use FFI\CData;
use Omegaalfa\Transformer\Exception\BackendException;
use Omegaalfa\Transformer\Tensor\Shape;
use Omegaalfa\Transformer\Tensor\Storage\NativeStorage;
use Omegaalfa\Transformer\Tensor\Tensor;
use Omegaalfa\Transformer\Transformer\AttentionMask;

final readonly class NativeLibrary
{
    private const STATUS_OK = 0;

    private FFI $ffi;

    public static function defaultPath(string $projectRoot, bool $release = true): string
    {
        $profile = $release ? 'release' : 'debug';
        $libraryName = match (PHP_OS_FAMILY) {
            'Darwin' => 'libtransformer_runtime.dylib',
            'Windows' => 'transformer_runtime.dll',
            default => 'libtransformer_runtime.so',
        };

        return rtrim($projectRoot, DIRECTORY_SEPARATOR)
            . "/runtime/target/{$profile}/{$libraryName}";
    }

    public function __construct(string $libraryPath)
    {
        if (!is_file($libraryPath) || !is_readable($libraryPath)) {
            throw new BackendException("Native library is not readable: {$libraryPath}");
        }

        $definitionsPath = __DIR__ . '/definitions.h';
        $definitions = file_get_contents($definitionsPath);

        if ($definitions === false) {
            throw new BackendException("Unable to read FFI definitions: {$definitionsPath}");
        }

        $this->ffi = FFI::cdef($definitions, $libraryPath);
    }

    public function version(): string
    {
        /** @var string|CData $version */
        // @phpstan-ignore method.notFound (Native methods are defined by FFI::cdef().)
        $version = $this->ffi->transformer_native_version();

        return is_string($version) ? $version : FFI::string($version);
    }

    /** @param list<float> $data */
    public function tensorFromFloat32(array $data, Shape $shape): Tensor
    {
        $dimensions = $shape->dimensions;
        $expected = $dimensions === [] ? 1 : array_product($dimensions);
        if ($expected !== count($data)) {
            throw new BackendException('Native tensor data length does not match its shape.');
        }

        $dataBuffer = $this->ffi->new('float[' . max(1, count($data)) . ']');
        foreach ($data as $index => $value) {
            // @phpstan-ignore offsetAccess.nonOffsetAccessible (FFI C array.)
            $dataBuffer[$index] = $value;
        }
        $rank = count($dimensions);
        $shapeBuffer = $this->ffi->new('size_t[' . max(1, $rank) . ']');
        foreach ($dimensions as $axis => $dimension) {
            // @phpstan-ignore offsetAccess.nonOffsetAccessible (FFI C array.)
            $shapeBuffer[$axis] = $dimension;
        }
        $output = $this->ffi->new('TransformerTensor *[1]');
        /** @var int $status */
        // @phpstan-ignore method.notFound (Native methods are defined by FFI::cdef().)
        $status = $this->ffi->transformer_tensor_create_f32($dataBuffer, $shapeBuffer, $rank, $output);
        if ($status !== self::STATUS_OK) {
            throw new BackendException("Native tensor creation failed with status {$status}.");
        }

        $storage = new NativeStorage($this->ffi, $output[0]);

        return new Tensor($storage->shape(), $storage);
    }

    /** @param list<int> $tokenIds */
    public function embeddingTokenIds(array $tokenIds, Shape $shape, Tensor $weight): Tensor
    {
        $storage = $weight->storage();
        if (!$storage instanceof NativeStorage) {
            throw new \LogicException('Embedding weight requires native storage.');
        }

        [$batch, $sequence] = $shape->dimensions;
        $outputStorage = $storage->embeddingTokenIds($tokenIds, $batch, $sequence, $this->ffi);

        return new Tensor($outputStorage->shape(), $outputStorage);
    }

    public function gelu(Tensor $input): Tensor
    {
        $storage = $input->storage();
        if (!$storage instanceof NativeStorage) {
            throw new \LogicException('GELU requires native storage.');
        }

        $outputStorage = $storage->gelu($this->ffi);

        return new Tensor($outputStorage->shape(), $outputStorage);
    }

    public function exactGelu(Tensor $input): Tensor
    {
        $storage = $input->storage();
        if (!$storage instanceof NativeStorage) {
            throw new \LogicException('ExactGELU requires native storage.');
        }

        $outputStorage = $storage->exactGelu($this->ffi);

        return new Tensor($outputStorage->shape(), $outputStorage);
    }

    public function bertSelfAttention(
        Tensor $input,
        Tensor $qWeight,
        Tensor $qBias,
        Tensor $kWeight,
        Tensor $kBias,
        Tensor $vWeight,
        Tensor $vBias,
        Tensor $outWeight,
        Tensor $outBias,
        int $heads,
        ?AttentionMask $mask = null,
    ): Tensor {
        $storages = [];
        foreach ([$input, $qWeight, $qBias, $kWeight, $kBias, $vWeight, $vBias, $outWeight, $outBias] as $tensor) {
            $storage = $tensor->storage();
            if (!$storage instanceof NativeStorage) {
                throw new \LogicException('BERT self-attention requires native storage.');
            }
            $storages[] = $storage;
        }

        [$inputStorage, $qWeightStorage, $qBiasStorage, $kWeightStorage, $kBiasStorage, $vWeightStorage, $vBiasStorage, $outWeightStorage, $outBiasStorage] = $storages;
        $outputStorage = $inputStorage->bertSelfAttention(
            $qWeightStorage,
            $qBiasStorage,
            $kWeightStorage,
            $kBiasStorage,
            $vWeightStorage,
            $vBiasStorage,
            $outWeightStorage,
            $outBiasStorage,
            $heads,
            $mask,
            $this->ffi,
        );

        return new Tensor($outputStorage->shape(), $outputStorage);
    }

    public function multiHeadAttention(
        Tensor $input,
        Tensor $qWeight,
        Tensor $kWeight,
        Tensor $vWeight,
        Tensor $outWeight,
        int $heads,
        ?AttentionMask $mask = null,
    ): Tensor {
        $storages = [];
        foreach ([$input, $qWeight, $kWeight, $vWeight, $outWeight] as $tensor) {
            $storage = $tensor->storage();
            if (!$storage instanceof NativeStorage) {
                throw new \LogicException('Multi-head attention requires native storage.');
            }
            $storages[] = $storage;
        }
        [$inputStorage, $qStorage, $kStorage, $vStorage, $outStorage] = $storages;
        $outputStorage = $inputStorage->multiHeadAttention(
            $qStorage,
            $kStorage,
            $vStorage,
            $outStorage,
            $heads,
            $mask,
            $this->ffi,
        );

        return new Tensor($outputStorage->shape(), $outputStorage);
    }

    /**
     * @param list<float> $a
     * @param list<float> $b
     * @return list<float>
     */
    public function addFloat32(array $a, array $b): array
    {
        $length = count($a);

        if ($length === 0) {
            return [];
        }

        /** @var CData $aBuffer */
        $aBuffer = $this->ffi->new("float[{$length}]");
        /** @var CData $bBuffer */
        $bBuffer = $this->ffi->new("float[{$length}]");
        /** @var CData $outputBuffer */
        $outputBuffer = $this->ffi->new("float[{$length}]");

        for ($index = 0; $index < $length; ++$index) {
            // @phpstan-ignore offsetAccess.nonOffsetAccessible (FFI C array.)
            $aBuffer[$index] = $a[$index];
            // @phpstan-ignore offsetAccess.nonOffsetAccessible (FFI C array.)
            $bBuffer[$index] = $b[$index];
        }

        /** @var int $status */
        // @phpstan-ignore method.notFound (Native methods are defined by FFI::cdef().)
        $status = $this->ffi->transformer_tensor_add_f32(
            $aBuffer,
            $bBuffer,
            $outputBuffer,
            $length,
        );

        if ($status !== self::STATUS_OK) {
            throw new BackendException("Native float32 addition failed with status {$status}.");
        }

        $output = [];

        for ($index = 0; $index < $length; ++$index) {
            // @phpstan-ignore cast.double (FFI C array element is a float.)
            $output[] = (float) $outputBuffer[$index];
        }

        return $output;
    }

    /**
     * @param list<float> $a Row-major M x K matrix.
     * @param list<float> $b Row-major K x N matrix.
     * @return list<float> Row-major M x N matrix.
     */
    public function matmulFloat32(array $a, array $b, int $m, int $k, int $n): array
    {
        $aLength = count($a);
        $bLength = count($b);
        $outputLength = $m * $n;

        if ($outputLength === 0) {
            return [];
        }

        /** @var CData $aBuffer */
        $aBuffer = $this->ffi->new('float[' . max(1, $aLength) . ']');
        /** @var CData $bBuffer */
        $bBuffer = $this->ffi->new('float[' . max(1, $bLength) . ']');
        /** @var CData $outputBuffer */
        $outputBuffer = $this->ffi->new("float[{$outputLength}]");

        for ($index = 0; $index < $aLength; ++$index) {
            // @phpstan-ignore offsetAccess.nonOffsetAccessible (FFI C array.)
            $aBuffer[$index] = $a[$index];
        }

        for ($index = 0; $index < $bLength; ++$index) {
            // @phpstan-ignore offsetAccess.nonOffsetAccessible (FFI C array.)
            $bBuffer[$index] = $b[$index];
        }

        /** @var int $status */
        // @phpstan-ignore method.notFound (Native methods are defined by FFI::cdef().)
        $status = $this->ffi->transformer_matmul_f32(
            $aBuffer,
            $bBuffer,
            $outputBuffer,
            $m,
            $k,
            $n,
        );

        if ($status !== self::STATUS_OK) {
            throw new BackendException("Native float32 matmul failed with status {$status}.");
        }

        $output = [];

        for ($index = 0; $index < $outputLength; ++$index) {
            // @phpstan-ignore cast.double (FFI C array element is a float.)
            $output[] = (float) $outputBuffer[$index];
        }

        return $output;
    }

    /**
     * @param list<float> $input Row-major rows x columns matrix.
     * @return list<float> Row-major columns x rows matrix.
     */
    public function transposeFloat32(array $input, int $rows, int $columns): array
    {
        $length = count($input);

        if ($length === 0) {
            return [];
        }

        /** @var CData $inputBuffer */
        $inputBuffer = $this->ffi->new("float[{$length}]");
        /** @var CData $outputBuffer */
        $outputBuffer = $this->ffi->new("float[{$length}]");

        for ($index = 0; $index < $length; ++$index) {
            // @phpstan-ignore offsetAccess.nonOffsetAccessible (FFI C array.)
            $inputBuffer[$index] = $input[$index];
        }

        /** @var int $status */
        // @phpstan-ignore method.notFound (Native methods are defined by FFI::cdef().)
        $status = $this->ffi->transformer_transpose_f32(
            $inputBuffer,
            $outputBuffer,
            $rows,
            $columns,
        );

        if ($status !== self::STATUS_OK) {
            throw new BackendException("Native float32 transpose failed with status {$status}.");
        }

        $output = [];

        for ($index = 0; $index < $length; ++$index) {
            // @phpstan-ignore cast.double (FFI C array element is a float.)
            $output[] = (float) $outputBuffer[$index];
        }

        return $output;
    }

    /**
     * @param non-empty-list<float> $input
     * @return non-empty-list<float>
     */
    public function softmaxFloat32(array $input): array
    {
        $length = count($input);
        /** @var CData $inputBuffer */
        $inputBuffer = $this->ffi->new("float[{$length}]");
        /** @var CData $outputBuffer */
        $outputBuffer = $this->ffi->new("float[{$length}]");

        for ($index = 0; $index < $length; ++$index) {
            // @phpstan-ignore offsetAccess.nonOffsetAccessible (FFI C array.)
            $inputBuffer[$index] = $input[$index];
        }

        /** @var int $status */
        // @phpstan-ignore method.notFound (Native methods are defined by FFI::cdef().)
        $status = $this->ffi->transformer_softmax_f32(
            $inputBuffer,
            $outputBuffer,
            $length,
        );

        if ($status !== self::STATUS_OK) {
            throw new BackendException("Native float32 softmax failed with status {$status}.");
        }

        $output = [];

        for ($index = 0; $index < $length; ++$index) {
            // @phpstan-ignore cast.double (FFI C array element is a float.)
            $output[] = (float) $outputBuffer[$index];
        }

        return $output;
    }
}
