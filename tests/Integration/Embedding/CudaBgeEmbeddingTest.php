<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tests\Integration\Embedding;

use Omegaalfa\Transformer\Backend\BackendType;
use Omegaalfa\Transformer\Backend\Ffi\FfiBackend;
use Omegaalfa\Transformer\Backend\Ffi\NativeLibrary;
use Omegaalfa\Transformer\Embedding\CudaBgeEmbeddingModel;
use Omegaalfa\Transformer\Model\Loader\CudaBgeEmbeddingModelLoader;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Runtime\RuntimeConfig;
use PHPUnit\Framework\TestCase;

final class CudaBgeEmbeddingTest extends TestCase
{
    private static ?CudaBgeEmbeddingModel $model = null;
    private static string $reference = '';

    public static function setUpBeforeClass(): void
    {
        $checkpoint = getenv('TRANSFORMER_BGE_CUDA_CHECKPOINT');
        if (!is_string($checkpoint) || $checkpoint === '') {
            return;
        }
        $root = dirname(__DIR__, 3);
        $library = NativeLibrary::defaultPath($root);
        $runtime = new Runtime(
            new FfiBackend(new NativeLibrary($library)),
            new RuntimeConfig(BackendType::Ffi),
        );
        self::$model = (new CudaBgeEmbeddingModelLoader($runtime, $library))->load($checkpoint);
        self::$reference = dirname($checkpoint) . '/reference/hello_world/cls_normalized.f32';
    }

    public function testRealCudaModelIsResidentDeterministicAndMatchesOfficialReference(): void
    {
        if (self::$model === null) {
            self::markTestSkipped('Set TRANSFORMER_BGE_CUDA_CHECKPOINT and build runtime with --features cuda.');
        }
        $identity = self::$model->library->identity();
        $first = self::$model->encode('hello world');
        $second = self::$model->encode('hello world');
        self::assertCount(384, $first);
        self::assertSame($first, $second);
        self::assertSame(197, self::$model->library->parameterCount());
        self::assertSame($identity, self::$model->library->identity());

        self::$model->library->setGraphEnabled(false);
        $ordinary = self::$model->encode('hello world');
        self::$model->library->setGraphEnabled(true);
        self::$model->encode('hello world');
        $graphed = self::$model->encode('hello world');
        self::$model->encode('hello world');
        $diagnostics = self::$model->library->benchmarkDiagnostics();
        self::assertSame($ordinary, $graphed);
        self::assertTrue($diagnostics['graph_enabled']);
        self::assertTrue($diagnostics['graph_ready']);
        self::assertTrue($diagnostics['graph_reused']);
        self::assertSame(1, $diagnostics['host_submissions']);
        self::assertSame(195, $diagnostics['internal_submissions']);

        $decoded = unpack('g*', (string) file_get_contents(self::$reference));
        self::assertIsArray($decoded);
        foreach (array_values($decoded) as $index => $expected) {
            self::assertIsFloat($expected);
            self::assertLessThanOrEqual(2.0e-5 + 2.0e-5 * abs($expected), abs($first[$index] - $expected));
        }

        $batch = self::$model->encodeBatch(['hello world', 'A second sentence.']);
        self::assertCount(2, $batch);
        self::assertCount(384, $batch[0]);
        self::assertCount(384, $batch[1]);
    }
}
