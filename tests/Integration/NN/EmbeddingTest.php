<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tests\Integration\NN;

use InvalidArgumentException;
use LogicException;
use Omegaalfa\Transformer\Backend\BackendType;
use Omegaalfa\Transformer\Backend\Ffi\FfiBackend;
use Omegaalfa\Transformer\Backend\Ffi\NativeLibrary;
use Omegaalfa\Transformer\NN\Embedding;
use Omegaalfa\Transformer\NN\Module;
use Omegaalfa\Transformer\NN\Parameter;
use Omegaalfa\Transformer\NN\TensorModule;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Runtime\RuntimeConfig;
use Omegaalfa\Transformer\Tensor\Shape;
use Omegaalfa\Transformer\Tensor\Tensor;
use PHPUnit\Framework\TestCase;

final class EmbeddingTest extends TestCase
{
    private FfiBackend $backend;
    private Runtime $runtime;
    private string $libraryPath;

    protected function setUp(): void
    {
        $this->libraryPath = NativeLibrary::defaultPath(dirname(__DIR__, 3));
        if (!is_file($this->libraryPath)) {
            self::markTestSkipped('Release native runtime is not built.');
        }
        $this->backend = new FfiBackend(new NativeLibrary($this->libraryPath));
        $this->runtime = new Runtime($this->backend, new RuntimeConfig(BackendType::Ffi));
    }

    public function testModuleContractsSeparateIntrospectionFromTensorForward(): void
    {
        $embedding = $this->embedding();
        $interfaces = class_implements($embedding);

        self::assertInstanceOf(Module::class, $embedding);
        self::assertContains(Module::class, $interfaces);
        self::assertNotContains(TensorModule::class, $interfaces);
        self::assertSame(['weight' => $embedding->weight], $embedding->parameters());
        self::assertSame([], $embedding->modules());
    }

    public function testRejectsInvalidDimensionsAndWeightShape(): void
    {
        $weight = new Parameter('weight', $this->tensor([1.0], [1, 1]));
        foreach ([[0, 1], [1, 0]] as [$vocabularySize, $dimensions]) {
            try {
                new Embedding($vocabularySize, $dimensions, $this->runtime, $weight);
                self::fail('Non-positive Embedding dimensions must fail.');
            } catch (InvalidArgumentException) {
            }
        }

        $this->expectException(InvalidArgumentException::class);
        new Embedding(2, 2, $this->runtime, $weight);
    }

    public function testGathersSingleSequenceAndBatchBitwise(): void
    {
        $embedding = $this->embedding();
        $residentWeight = $embedding->weight->tensor->toFloat32();

        $single = $embedding->forwardTokenIds([0, 3, 1], new Shape([1, 3]));
        self::assertSame([1, 3, 3], $single->shape()->dimensions);
        self::assertSame($this->reference([0, 3, 1], $residentWeight, 3), $single->toFloat32());

        $batch = $embedding->forwardTokenIds([3, 0, 2, 1], new Shape([2, 2]));
        self::assertSame([2, 2, 3], $batch->shape()->dimensions);
        self::assertSame($this->reference([3, 0, 2, 1], $residentWeight, 3), $batch->toFloat32());
    }

    public function testSupportsBoundaryAndRepeatedTokens(): void
    {
        $embedding = $this->embedding();
        $weight = $embedding->weight->tensor->toFloat32();
        $ids = [0, 3, 3, 0];

        self::assertSame(
            $this->reference($ids, $weight, 3),
            $embedding->forwardTokenIds($ids, new Shape([2, 2]))->toFloat32(),
        );
    }

    public function testRejectsInvalidShapeStructureAndTokenCount(): void
    {
        $embedding = $this->embedding();
        foreach (
            [
                [[0], new Shape([1])],
                [[0], new Shape([1, 1, 1])],
                [[0], new Shape([-1, 1])],
                [[0], new Shape([1, -1])],
                [[0], new Shape([1, 2])],
                [[], new Shape([PHP_INT_MAX, 2])],
            ] as [$ids, $shape]
        ) {
            try {
                $embedding->forwardTokenIds($ids, $shape);
                self::fail('Invalid Embedding shape or token count must fail.');
            } catch (InvalidArgumentException) {
            }
        }
        self::addToAssertionCount(1);
    }

    public function testRejectsInvalidTokenListsAndRanges(): void
    {
        $embedding = $this->embedding();
        foreach (
            [
                [0 => 0, 2 => 1],
                [0, 1.0],
                [0, '1'],
                [0, -1],
                [0, 4],
            ] as $ids
        ) {
            try {
                $this->invokeWithUntrustedIds($embedding, $ids, new Shape([1, 2]));
                self::fail('Invalid Embedding token IDs must fail.');
            } catch (InvalidArgumentException) {
            }
        }
        self::addToAssertionCount(1);
    }

    public function testSupportsAllApprovedEmptyShapes(): void
    {
        $embedding = $this->embedding();
        foreach ([[0, 3], [2, 0], [0, 0]] as $shape) {
            $output = $embedding->forwardTokenIds([], new Shape($shape));
            self::assertSame([...$shape, 3], $output->shape()->dimensions);
            self::assertSame([], $output->toFloat32());
        }
    }

    public function testWeightStaysResidentAndOutputsRemainIndependent(): void
    {
        $embedding = $this->embedding();
        $weightBefore = $embedding->weight->tensor->toFloat32();
        $storageId = spl_object_id($embedding->weight->tensor->storage());
        $first = $embedding->forwardTokenIds([0, 1], new Shape([1, 2]));
        $firstValues = $first->toFloat32();

        for ($iteration = 0; $iteration < 10; ++$iteration) {
            $ids = [$iteration % 4, ($iteration + 1) % 4];
            $current = $embedding->forwardTokenIds($ids, new Shape([1, 2]));
            self::assertSame($this->reference($ids, $weightBefore, 3), $current->toFloat32());
        }

        self::assertSame($storageId, spl_object_id($embedding->weight->tensor->storage()));
        self::assertSame($weightBefore, $embedding->weight->tensor->toFloat32());
        self::assertSame($firstValues, $first->toFloat32());
        self::assertNotSame(
            $firstValues,
            $embedding->forwardTokenIds([2, 3], new Shape([1, 2]))->toFloat32(),
        );
    }

    public function testInvalidCallDoesNotConsumeResidentWeight(): void
    {
        $embedding = $this->embedding();
        $expected = $embedding->forwardTokenIds([1], new Shape([1, 1]))->toFloat32();

        try {
            $embedding->forwardTokenIds([-1], new Shape([1, 1]));
            self::fail('Invalid token must fail.');
        } catch (InvalidArgumentException) {
            self::assertSame($expected, $embedding->forwardTokenIds([1], new Shape([1, 1]))->toFloat32());
        }
    }

    public function testIndependentInstancesDoNotShareOrInvalidateWeights(): void
    {
        $embeddingA = $this->embedding();
        $weightB = new Parameter('weight', $this->tensor(array_fill(0, 12, 7.0), [4, 3]));
        $embeddingB = new Embedding(4, 3, $this->runtime, $weightB);
        $expectedB = $embeddingB->forwardTokenIds([0, 3], new Shape([1, 2]))->toFloat32();

        self::assertNotSame(
            spl_object_id($embeddingA->weight->tensor->storage()),
            spl_object_id($embeddingB->weight->tensor->storage()),
        );
        $embeddingA->weight->tensor->destroy();
        self::assertSame($expectedB, $embeddingB->forwardTokenIds([0, 3], new Shape([1, 2]))->toFloat32());
    }

    public function testDestroyedWeightAndDifferentRuntimeFailCleanly(): void
    {
        $destroyed = $this->embedding();
        $destroyed->weight->tensor->destroy();
        try {
            $destroyed->forwardTokenIds([0], new Shape([1, 1]));
            self::fail('Destroyed weight must fail.');
        } catch (LogicException) {
        }

        $otherBackend = new FfiBackend(new NativeLibrary($this->libraryPath));
        $otherRuntime = new Runtime($otherBackend, new RuntimeConfig(BackendType::Ffi));
        $foreign = new Embedding(4, 3, $otherRuntime, $this->embedding()->weight);
        $this->expectException(LogicException::class);
        $foreign->forwardTokenIds([0], new Shape([1, 1]));
    }

    private function embedding(): Embedding
    {
        $values = [
            0.0, 0.1, 0.2,
            1.0, 1.1, 1.2,
            2.0, 2.1, 2.2,
            3.0, 3.1, 3.2,
        ];
        $weight = new Parameter('weight', $this->tensor($values, [4, 3]));

        return new Embedding(4, 3, $this->runtime, $weight);
    }

    /**
     * @param list<float|int> $values
     * @param list<int> $shape
     */
    private function tensor(array $values, array $shape): Tensor
    {
        return $this->backend->tensorFromFloat32($values, new Shape($shape));
    }

    /**
     * @param list<int> $tokenIds
     * @param list<float> $weight
     * @return list<float>
     */
    private function reference(array $tokenIds, array $weight, int $embeddingDim): array
    {
        $output = [];
        foreach ($tokenIds as $tokenId) {
            array_push($output, ...array_slice($weight, $tokenId * $embeddingDim, $embeddingDim));
        }

        return $output;
    }

    /** @param array<array-key, mixed> $tokenIds */
    private function invokeWithUntrustedIds(Embedding $embedding, array $tokenIds, Shape $shape): mixed
    {
        return (new \ReflectionMethod($embedding, 'forwardTokenIds'))->invoke($embedding, $tokenIds, $shape);
    }
}
