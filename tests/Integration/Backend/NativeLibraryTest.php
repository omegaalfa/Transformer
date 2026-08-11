<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tests\Integration\Backend;

use InvalidArgumentException;
use Omegaalfa\Transformer\Backend\Ffi\NativeLibrary;
use Omegaalfa\Transformer\Backend\Ffi\FfiBackend;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;

#[CoversClass(NativeLibrary::class)]
final class NativeLibraryTest extends TestCase
{
    public function testReadsVersionFromCompiledNativeRuntime(): void
    {
        if (!extension_loaded('FFI')) {
            self::markTestSkipped('The PHP FFI extension is not loaded.');
        }

        $libraryPath = NativeLibrary::defaultPath(dirname(__DIR__, 3));

        if (!is_file($libraryPath)) {
            self::markTestSkipped('The native runtime artifact has not been built.');
        }

        self::assertSame('0.1.0-dev', (new NativeLibrary($libraryPath))->version());
    }

    /** @return iterable<string, array{list<int|float>, list<int|float>, list<float>}> */
    public static function additionCases(): iterable
    {
        yield 'positive integers' => [[1, 2, 3], [10, 20, 30], [11.0, 22.0, 33.0]];
        yield 'fractions and negatives' => [[1.5, -2.25, 0.5], [2.5, 1.25, 4.0], [4.0, -1.0, 4.5]];
        yield 'cancellation' => [[-1, -2, 3], [1, 5, -3], [0.0, 3.0, 0.0]];
        yield 'empty buffers' => [[], [], []];
    }

    /**
     * @param list<int|float> $a
     * @param list<int|float> $b
     * @param list<float> $expected
     */
    #[DataProvider('additionCases')]
    public function testAddsFloat32Buffers(array $a, array $b, array $expected): void
    {
        $libraryPath = NativeLibrary::defaultPath(dirname(__DIR__, 3));

        if (!extension_loaded('FFI')) {
            self::markTestSkipped('The PHP FFI extension is not loaded.');
        }

        if (!is_file($libraryPath)) {
            self::markTestSkipped('The Rust runtime artifact has not been built.');
        }

        $backend = new FfiBackend(new NativeLibrary($libraryPath));

        self::assertSame($expected, $backend->addFloat32($a, $b));
    }

    /**
     * @return iterable<string, array{
     *     list<int|float>,
     *     list<int|float>,
     *     int,
     *     int,
     *     int,
     *     list<float>
     * }>
     */
    public static function matmulCases(): iterable
    {
        yield '2x3 by 3x2' => [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            2,
            3,
            2,
            [58.0, 64.0, 139.0, 154.0],
        ];
        yield 'row by column' => [[1, 2, 3], [4, 5, 6], 1, 3, 1, [32.0]];
        yield 'identity matrix' => [[2, -1, 3, 4], [1, 0, 0, 1], 2, 2, 2, [2.0, -1.0, 3.0, 4.0]];
        yield 'zero inner dimension' => [[], [], 2, 0, 2, [0.0, 0.0, 0.0, 0.0]];
    }

    /**
     * @param list<int|float> $a
     * @param list<int|float> $b
     * @param list<float> $expected
     */
    #[DataProvider('matmulCases')]
    public function testMultipliesFloat32Matrices(
        array $a,
        array $b,
        int $m,
        int $k,
        int $n,
        array $expected,
    ): void {
        $libraryPath = NativeLibrary::defaultPath(dirname(__DIR__, 3));

        if (!extension_loaded('FFI')) {
            self::markTestSkipped('The PHP FFI extension is not loaded.');
        }

        if (!is_file($libraryPath)) {
            self::markTestSkipped('The Rust runtime artifact has not been built.');
        }

        $backend = new FfiBackend(new NativeLibrary($libraryPath));

        self::assertSame($expected, $backend->matmulFloat32($a, $b, $m, $k, $n));
    }

    /** @return iterable<string, array{list<int|float>, int, int, list<float>}> */
    public static function transposeCases(): iterable
    {
        yield '2x3 to 3x2' => [[1, 2, 3, 4, 5, 6], 2, 3, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]];
        yield 'row to column' => [[1, 2, 3], 1, 3, [1.0, 2.0, 3.0]];
        yield 'square matrix' => [[1, 2, 3, 4], 2, 2, [1.0, 3.0, 2.0, 4.0]];
        yield 'empty shape' => [[], 0, 3, []];
    }

    /**
     * @param list<int|float> $input
     * @param list<float> $expected
     */
    #[DataProvider('transposeCases')]
    public function testTransposesFloat32Matrices(
        array $input,
        int $rows,
        int $columns,
        array $expected,
    ): void {
        $libraryPath = NativeLibrary::defaultPath(dirname(__DIR__, 3));

        if (!extension_loaded('FFI')) {
            self::markTestSkipped('The PHP FFI extension is not loaded.');
        }

        if (!is_file($libraryPath)) {
            self::markTestSkipped('The Rust runtime artifact has not been built.');
        }

        $backend = new FfiBackend(new NativeLibrary($libraryPath));

        self::assertSame($expected, $backend->transposeFloat32($input, $rows, $columns));
    }

    /** @return iterable<string, array{list<int|float>, list<float>}> */
    public static function softmaxCases(): iterable
    {
        yield 'reference values' => [[1, 2, 3], [0.090_030_57, 0.244_728_48, 0.665_240_94]];
        yield 'large positive values' => [[1000, 1001, 1002], [0.090_030_57, 0.244_728_48, 0.665_240_94]];
        yield 'uniform values' => [[0, 0, 0], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]];
        yield 'large negative values' => [[-1000, -1001, -1002], [0.665_240_94, 0.244_728_48, 0.090_030_57]];
    }

    /**
     * @param list<int|float> $input
     * @param list<float> $expected
     */
    #[DataProvider('softmaxCases')]
    public function testComputesStableFloat32Softmax(array $input, array $expected): void
    {
        $libraryPath = NativeLibrary::defaultPath(dirname(__DIR__, 3));

        if (!extension_loaded('FFI')) {
            self::markTestSkipped('The PHP FFI extension is not loaded.');
        }

        if (!is_file($libraryPath)) {
            self::markTestSkipped('The Rust runtime artifact has not been built.');
        }

        $backend = new FfiBackend(new NativeLibrary($libraryPath));
        $actual = $backend->softmaxFloat32($input);

        foreach ($expected as $index => $value) {
            self::assertEqualsWithDelta($value, $actual[$index], 1.0e-5);
            self::assertTrue(is_finite($actual[$index]));
        }

        self::assertEqualsWithDelta(1.0, array_sum($actual), 1.0e-5);
    }

    public function testSoftmaxRejectsEmptyInputBeforeFfiCall(): void
    {
        $libraryPath = NativeLibrary::defaultPath(dirname(__DIR__, 3));

        if (!is_file($libraryPath)) {
            self::markTestSkipped('The Rust runtime artifact has not been built.');
        }

        $backend = new FfiBackend(new NativeLibrary($libraryPath));

        $this->expectException(InvalidArgumentException::class);
        $backend->softmaxFloat32([]);
    }
}
