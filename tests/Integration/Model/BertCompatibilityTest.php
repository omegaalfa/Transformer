<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tests\Integration\Model;

use InvalidArgumentException;
use Omegaalfa\Transformer\Backend\BackendType;
use Omegaalfa\Transformer\Backend\Ffi\FfiBackend;
use Omegaalfa\Transformer\Backend\Ffi\NativeLibrary;
use Omegaalfa\Transformer\Exception\BackendException;
use Omegaalfa\Transformer\Model\Encoder\Bert\BertConfig;
use Omegaalfa\Transformer\Model\Encoder\Bert\BertEmbeddings;
use Omegaalfa\Transformer\Model\Encoder\Bert\BertFeedForward;
use Omegaalfa\Transformer\Model\Encoder\Bert\BertModel;
use Omegaalfa\Transformer\Model\Encoder\Bert\BertModelFactory;
use Omegaalfa\Transformer\Model\Encoder\Bert\BertModelInput;
use Omegaalfa\Transformer\Model\Encoder\Bert\BertSelfAttention;
use Omegaalfa\Transformer\Model\Encoder\Bert\BertTransformerBlock;
use Omegaalfa\Transformer\NN\Activation\ExactGelu;
use Omegaalfa\Transformer\NN\Embedding;
use Omegaalfa\Transformer\NN\LayerNorm;
use Omegaalfa\Transformer\NN\Linear;
use Omegaalfa\Transformer\NN\Parameter;
use Omegaalfa\Transformer\Exception\ModelException;
use Omegaalfa\Transformer\Model\Loader\BgeSmallEnV15Manifest;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Runtime\RuntimeConfig;
use Omegaalfa\Transformer\Tensor\Shape;
use Omegaalfa\Transformer\Tensor\Tensor;
use Omegaalfa\Transformer\Transformer\AttentionMask;
use PHPUnit\Framework\TestCase;

final class BertCompatibilityTest extends TestCase
{
    private FfiBackend $backend;
    private Runtime $runtime;

    protected function setUp(): void
    {
        $path = NativeLibrary::defaultPath(dirname(__DIR__, 3));
        if (!is_file($path)) {
            self::markTestSkipped('Release native runtime is not built.');
        }
        $this->backend = new FfiBackend(new NativeLibrary($path));
        $this->runtime = new Runtime($this->backend, new RuntimeConfig(BackendType::Ffi));
    }

    public function testExactGeluMatchesIndependentErfReferenceAndPreservesShape(): void
    {
        $values = [-10.0, -1.0, -0.0001, -0.0, 0.0001, 1.0, 10.0];
        $input = $this->tensor($values, [1, 7]);
        $output = (new ExactGelu($this->runtime))->forward($input);
        self::assertSame([1, 7], $output->shape()->dimensions);
        $expectedValues = [0.0, -0.1586552539, -0.0000499960, -0.0, 0.0000500040, 0.8413447461, 10.0];
        foreach ($output->toFloat32() as $index => $actual) {
            $expected = $expectedValues[$index];
            self::assertLessThanOrEqual(1e-6 + 1e-6 * abs($expected), abs($actual - $expected));
        }
        self::assertTrue($output->toFloat32()[3] === 0.0);

        foreach ([NAN, INF, -INF] as $invalid) {
            try {
                (new ExactGelu($this->runtime))->forward($this->tensor([$invalid], [1]));
                self::fail('ExactGELU accepted a non-finite input.');
            } catch (BackendException) {
                self::addToAssertionCount(1);
            }
        }

        self::assertEqualsWithDelta(
            [0.8413447141647339],
            (new ExactGelu($this->runtime))->forward($this->tensor([1.0], [1]))->toFloat32(),
            1e-6,
        );
    }

    public function testBertAttentionUsesBiasMaskAndResidentParameters(): void
    {
        $attention = $this->attention($this->config(1));
        $input = $this->tensor([1, 2, 3, 4], [1, 2, 2]);
        $mask = new AttentionMask([true, false], new Shape([1, 2]));
        $first = $attention->forward($input, $mask);
        self::assertSame([1, 2, 2], $first->shape()->dimensions);
        self::assertEqualsWithDelta([1.25, 1.75, 1.25, 1.75], $first->toFloat32(), 1e-6);
        self::assertSame($first->toFloat32(), $attention->forward($input, $mask)->toFloat32());
        foreach ($attention->modules() as $projection) {
            self::assertInstanceOf(Linear::class, $projection);
            self::assertNotNull($projection->bias);
        }

        $this->expectException(InvalidArgumentException::class);
        $attention->forward($input, new AttentionMask([true], new Shape([1, 1])));
    }

    public function testPostNormOrderingMatchesExplicitComposition(): void
    {
        $config = $this->config(1);
        $block = $this->block($config);
        $input = $this->tensor([0.2, -0.4, 0.8, 0.1], [1, 2, 2]);
        $attention = $block->attention->forward($input);
        $a = $block->attentionNorm->forward($input->add($attention));
        $ffn = $block->feedForward->forward($a);
        $expected = $block->feedForwardNorm->forward($a->add($ffn))->toFloat32();
        self::assertSame($expected, $block->forward($input)->toFloat32());
    }

    public function testBertAttentionRejectsFullyMaskedRowsWithoutBreakingLaterForward(): void
    {
        $attention = $this->attention($this->config(1));
        $input = $this->tensor([1, 2, 3, 4], [1, 2, 2]);

        try {
            $attention->forward($input, new AttentionMask([false, false], new Shape([1, 2])));
            self::fail('BERT Attention accepted a fully masked row.');
        } catch (BackendException) {
            self::addToAssertionCount(1);
        }

        self::assertSame([1, 2, 2], $attention->forward($input)->shape()->dimensions);
    }

    public function testEmbeddingsDerivePositionsDefaultTokenTypesAndValidateLimits(): void
    {
        $config = $this->config(1);
        $embeddings = $this->embeddings($config);
        $shape = new Shape([1, 2]);
        $default = $embeddings->forward([1, 2], $shape)->toFloat32();
        self::assertSame($default, $embeddings->forward([1, 2], $shape, [0, 0])->toFloat32());
        self::assertNotSame($default, $embeddings->forward([1, 2], $shape, [1, 1])->toFloat32());
        foreach ([[[1, 2, 3], new Shape([1, 3]), null], [[1], new Shape([1, 1]), [2]]] as [$ids, $invalidShape, $types]) {
            try {
                $embeddings->forward($ids, $invalidShape, $types);
                self::fail('Invalid BERT embedding input accepted.');
            } catch (InvalidArgumentException) {
                self::addToAssertionCount(1);
            }
        }
    }

    public function testModelRunsMultipleLayersRepeatedlyAndInstancesAreIndependent(): void
    {
        $config = $this->config(2);
        $a = $this->model($config);
        $b = $this->model($config);
        $input = new BertModelInput([1, 2, 3, 4], new Shape([2, 2]), new AttentionMask([true, true, true, false], new Shape([2, 2])), [0, 1, 0, 1]);
        $first = $a->forward($input)->lastHiddenState;
        self::assertSame([2, 2, 2], $first->shape()->dimensions);
        self::assertCount(2, $a->layers);
        self::assertSame($first->toFloat32(), $a->forward($input)->lastHiddenState->toFloat32());
        self::assertSame($first->toFloat32(), $b->forward($input)->lastHiddenState->toFloat32());
        self::assertNotSame(spl_object_id($a->embeddings->word->weight->tensor->storage()), spl_object_id($b->embeddings->word->weight->tensor->storage()));
    }

    public function testFactoryPublishesOnlyACompleteParameterTree(): void
    {
        $config = $this->config(1);
        $parameters = [];
        foreach (BgeSmallEnV15Manifest::create($config)->parameters as $spec) {
            $shape = $spec->materialization->runtimeShape->dimensions;
            $length = 1;
            foreach ($shape as $dimension) {
                $length *= $dimension;
            }
            $values = str_ends_with($spec->parameterName, '.weight') && count($shape) === 1
                ? array_fill(0, $length, 1)
                : array_fill(0, $length, 0);
            $parameters[$spec->parameterName] = $this->parameter($spec->parameterName, $values, $shape);
        }
        $missing = $parameters;
        array_pop($missing);
        try {
            (new BertModelFactory($this->runtime))->create($config, $missing);
            self::fail('Incomplete BERT Parameter tree was published.');
        } catch (ModelException) {
            self::addToAssertionCount(1);
        }
        $model = (new BertModelFactory($this->runtime))->create($config, $parameters);
        self::assertCount(1, $model->layers);
        self::assertSame([1, 1, 2], $model->forward(new BertModelInput([1], new Shape([1, 1])))->lastHiddenState->shape()->dimensions);
    }

    private function config(int $layers): BertConfig
    {
        return new BertConfig(6, 2, 3, 1, $layers, 2, 2, 1e-5);
    }

    private function model(BertConfig $config): BertModel
    {
        $layers = [];
        for ($index = 0; $index < $config->numHiddenLayers; ++$index) {
            $layers[] = $this->block($config);
        }
        return new BertModel($config, $this->runtime, $this->embeddings($config), $layers);
    }

    private function embeddings(BertConfig $config): BertEmbeddings
    {
        $wordValues = [];
        for ($index = 0; $index < $config->vocabularySize * 2; ++$index) {
            $wordValues[] = (($index % 7) - 3) / 4;
        }
        return new BertEmbeddings(
            $config,
            $this->runtime,
            new Embedding($config->vocabularySize, 2, $this->runtime, $this->parameter('word', $wordValues, [$config->vocabularySize, 2])),
            new Embedding(2, 2, $this->runtime, $this->parameter('position', [0.1, 0.2, 0.3, 0.4], [2, 2])),
            new Embedding(2, 2, $this->runtime, $this->parameter('type', [0, 0, 0.5, -0.5], [2, 2])),
            $this->norm(2),
        );
    }

    private function block(BertConfig $config): BertTransformerBlock
    {
        return new BertTransformerBlock(
            $config,
            $this->runtime,
            $this->attention($config),
            $this->norm(2),
            new BertFeedForward(
                $config,
                $this->runtime,
                new Linear(2, 3, $this->runtime, $this->parameter('wi', [0.2, -0.1, 0.3, 0.4, 0.1, -0.2], [2, 3]), $this->parameter('bi', [0.1, -0.2, 0.3], [3])),
                new ExactGelu($this->runtime),
                new Linear(3, 2, $this->runtime, $this->parameter('wo', [0.1, 0.3, -0.2, 0.4, 0.2, -0.1], [3, 2]), $this->parameter('bo', [0.05, -0.05], [2])),
            ),
            $this->norm(2),
        );
    }

    private function attention(BertConfig $config): BertSelfAttention
    {
        $identity = [1, 0, 0, 1];
        return new BertSelfAttention(
            $config,
            $this->runtime,
            new Linear(2, 2, $this->runtime, $this->parameter('qw', $identity, [2, 2]), $this->parameter('qb', [0.1, -0.1], [2])),
            new Linear(2, 2, $this->runtime, $this->parameter('kw', $identity, [2, 2]), $this->parameter('kb', [0, 0], [2])),
            new Linear(2, 2, $this->runtime, $this->parameter('vw', $identity, [2, 2]), $this->parameter('vb', [0, 0], [2])),
            new Linear(2, 2, $this->runtime, $this->parameter('ow', $identity, [2, 2]), $this->parameter('ob', [0.25, -0.25], [2])),
        );
    }

    private function norm(int $dimensions): LayerNorm
    {
        return new LayerNorm($dimensions, $this->runtime, $this->parameter('gamma', array_fill(0, $dimensions, 1), [$dimensions]), $this->parameter('beta', array_fill(0, $dimensions, 0), [$dimensions]));
    }

    /**
     * @param list<float|int> $values
     * @param list<int>       $shape
     */
    private function parameter(string $name, array $values, array $shape): Parameter
    {
        return new Parameter($name, $this->tensor($values, $shape), false);
    }

    /**
     * @param list<float|int> $values
     * @param list<int>       $shape
     */
    private function tensor(array $values, array $shape): Tensor
    {
        return $this->backend->tensorFromFloat32(array_map(static fn (float|int $value): float => (float) $value, $values), new Shape($shape));
    }
}
