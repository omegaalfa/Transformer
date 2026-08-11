<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Backend\Ffi;

use FFI;
use FFI\CData;
use Omegaalfa\Transformer\Exception\BackendException;

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
