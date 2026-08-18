<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tests\Integration\Embedding;

use InvalidArgumentException;
use Omegaalfa\Transformer\Backend\BackendType;
use Omegaalfa\Transformer\Backend\Ffi\FfiBackend;
use Omegaalfa\Transformer\Backend\Ffi\NativeLibrary;
use Omegaalfa\Transformer\Embedding\BgeEmbeddingModel;
use Omegaalfa\Transformer\Embedding\BgePoolingStrategy;
use Omegaalfa\Transformer\Embedding\Normalization\L2Normalizer;
use Omegaalfa\Transformer\Embedding\Pooling\ClsPooling;
use Omegaalfa\Transformer\Embedding\Pooling\MeanPooling;
use Omegaalfa\Transformer\Model\Encoder\Bert\BertModelInput;
use Omegaalfa\Transformer\Model\Encoder\Bert\BertModelInterface;
use Omegaalfa\Transformer\Model\ModelConfig;
use Omegaalfa\Transformer\Model\ModelOutput;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Runtime\RuntimeConfig;
use Omegaalfa\Transformer\Tensor\Shape;
use Omegaalfa\Transformer\Tensor\Tensor;
use Omegaalfa\Transformer\Tokenizer\BertTokenizer;
use Omegaalfa\Transformer\Transformer\Config\TransformerConfig;
use PHPUnit\Framework\TestCase;

final class BgeEmbeddingPipelineTest extends TestCase
{
    private Runtime $runtime;

    protected function setUp(): void
    {
        $path = NativeLibrary::defaultPath(dirname(__DIR__, 3));
        if (!is_file($path)) {
            self::markTestSkipped('Release native runtime is not built.');
        }
        $backend = new FfiBackend(new NativeLibrary($path));
        $this->runtime = new Runtime($backend, new RuntimeConfig(BackendType::Ffi));
    }

    public function testMaskedMeanPoolingMatchesManualRowsAndRejectsEmptyMaskRow(): void
    {
        $hidden = $this->tensor([1, 2, 3, 4, 100, 100, 0, 3, 4, 0, 0, 0], [2, 3, 2]);
        $mask = $this->tensor([1, 1, 0, 1, 1, 1], [2, 3]);
        $output = (new MeanPooling($this->runtime))->pool($hidden, $mask);
        self::assertSame([2, 2], $output->shape()->dimensions);
        self::assertEqualsWithDelta([2, 3, 4 / 3, 1], $output->toFloat32(), 1e-7);

        $this->expectException(InvalidArgumentException::class);
        (new MeanPooling($this->runtime))->pool(
            $this->tensor([1, 2], [1, 1, 2]),
            $this->tensor([0], [1, 1]),
        );
    }

    public function testClsPoolingSelectsPositionZeroForEveryBatchAndIgnoresPadding(): void
    {
        $hidden = $this->tensor([1, 2, 30, 40, 50, 60, -3, 4, 70, 80, 90, 100], [2, 3, 2]);
        $mask = $this->tensor([1, 1, 0, 1, 0, 0], [2, 3]);
        $output = (new ClsPooling($this->runtime))->pool($hidden, $mask);
        self::assertSame([2, 2], $output->shape()->dimensions);
        self::assertSame([1.0, 2.0, -3.0, 4.0], $output->toFloat32());

        $changedPadding = $this->tensor([1, 2, -300, 400, 500, -600, -3, 4, -700, 800, 900, -1000], [2, 3, 2]);
        self::assertSame($output->toFloat32(), (new ClsPooling($this->runtime))->pool($changedPadding)->toFloat32());
    }

    public function testL2NormalizationProducesUnitRowsAndRejectsZeroNorm(): void
    {
        $normalizer = new L2Normalizer($this->runtime);
        $output = $normalizer->normalize($this->tensor([3, 4, 5, 12], [2, 2]));
        self::assertEqualsWithDelta([0.6, 0.8, 5 / 13, 12 / 13], $output->toFloat32(), 1e-7);
        foreach (array_chunk($output->toFloat32(), 2) as $row) {
            self::assertEqualsWithDelta(1.0, sqrt($row[0] ** 2 + $row[1] ** 2), 1e-6);
        }

        $this->expectException(InvalidArgumentException::class);
        $normalizer->normalize($this->tensor([0, 0], [1, 2]));
    }

    public function testPublicPipelineSupportsSingleBatchPaddingRepeatAndIndependentOutputs(): void
    {
        $tokenizer = $this->tokenizer();
        self::assertSame([2, 4, 5, 3], $tokenizer->encode('Hello world')->tokenIds);
        self::assertSame([2, 4, 6, 5, 7, 3], $tokenizer->encode('Hello, world!')->tokenIds);
        $model = new RecordingBertModel($this->runtime);
        $pipeline = new BgeEmbeddingModel($model, $tokenizer, $this->runtime);
        $single = $pipeline->encode('hello world');
        self::assertCount(2, $single);

        $first = $pipeline->encodeBatch(['hello', 'hello world']);
        $second = $pipeline->encodeBatch(['hello', 'hello world']);
        self::assertCount(2, $first);
        self::assertCount(2, $first[0]);
        self::assertSame($first, $second);

        $firstOutput = $pipeline->encodeBatchOutput(['hello', 'hello world']);
        $secondOutput = $pipeline->encodeBatchOutput(['hello', 'hello world']);
        self::assertSame([2, 2], $firstOutput->embedding->shape()->dimensions);
        self::assertSame(BgePoolingStrategy::Cls, $pipeline->poolingStrategy);
        self::assertEqualsWithDelta([2.0, 1.0, 2.0, 1.0], $firstOutput->pooled->toFloat32(), 1e-7);
        self::assertSame($firstOutput->embedding->toFloat32(), $secondOutput->embedding->toFloat32());
        self::assertNotSame($firstOutput->embedding->storage(), $secondOutput->embedding->storage());
        self::assertNotNull($model->lastInput);
        self::assertNotNull($model->lastInput->attentionMask);
        self::assertSame([true, true, true, false, true, true, true, true], $model->lastInput->attentionMask->values);
        self::assertSame([0, 0, 0, 0, 0, 0, 0, 0], $model->lastInput->tokenTypeIds);

        $mean = new BgeEmbeddingModel($model, $tokenizer, $this->runtime, BgePoolingStrategy::Mean);
        $meanOutput = $mean->encodeBatchOutput(['hello', 'hello world']);
        self::assertSame(BgePoolingStrategy::Mean, $mean->poolingStrategy);
        self::assertNotSame($firstOutput->pooled->toFloat32(), $meanOutput->pooled->toFloat32());
    }

    private function tokenizer(): BertTokenizer
    {
        $path = tempnam(sys_get_temp_dir(), 'bert-tokenizer-');
        self::assertIsString($path);
        file_put_contents($path, json_encode([
            'model' => [
                'type' => 'WordPiece',
                'unk_token' => '[UNK]',
                'continuing_subword_prefix' => '##',
                'vocab' => ['[PAD]' => 0, '[UNK]' => 1, '[CLS]' => 2, '[SEP]' => 3, 'hello' => 4, 'world' => 5, ',' => 6, '!' => 7],
            ],
        ], JSON_THROW_ON_ERROR));
        try {
            return BertTokenizer::fromTokenizerJson($path, 8);
        } finally {
            @unlink($path);
        }
    }

    /**
     * @param list<float|int> $values
     * @param list<int>       $shape
     */
    private function tensor(array $values, array $shape): Tensor
    {
        return $this->runtime->backend()->tensorFromFloat32(
            array_map(static fn (float|int $value): float => (float) $value, $values),
            new Shape($shape),
        );
    }
}

final class RecordingBertModel implements BertModelInterface
{
    public ?BertModelInput $lastInput = null;

    public function __construct(private readonly Runtime $runtime)
    {
    }

    public function modelConfig(): ModelConfig
    {
        return new ModelConfig('BertModel', 8, new TransformerConfig(2, 1, 3, 1));
    }

    public function forward(BertModelInput $input): ModelOutput
    {
        $this->lastInput = $input;
        $values = [];
        foreach ($input->inputIds as $id) {
            $values[] = (float) $id;
            $values[] = 1.0;
        }
        return new ModelOutput($this->runtime->backend()->tensorFromFloat32(
            $values,
            new Shape([$input->shape->dimensions[0], $input->shape->dimensions[1], 2]),
        ));
    }
}
