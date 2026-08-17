<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\NN;

use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Tensor\Shape;
use Omegaalfa\Transformer\Tensor\Tensor;

final readonly class Embedding implements Module
{
    public function __construct(
        public int $vocabularySize,
        public int $dimensions,
        public Runtime $runtime,
        public Parameter $weight,
    ) {
        if ($vocabularySize <= 0 || $dimensions <= 0) {
            throw new \InvalidArgumentException('Embedding dimensions must be positive.');
        }
        if ($weight->tensor->shape()->dimensions !== [$vocabularySize, $dimensions]) {
            throw new \InvalidArgumentException('Embedding weight shape must be [vocabulary_size, embedding_dim].');
        }
    }

    /** @param list<int> $tokenIds */
    public function forwardTokenIds(array $tokenIds, Shape $shape): Tensor
    {
        $this->validateTokenIds($tokenIds, $shape);

        return $this->runtime->backend()->embeddingTokenIds($tokenIds, $shape, $this->weight->tensor);
    }

    public function parameters(): array
    {
        return ['weight' => $this->weight];
    }

    public function modules(): array
    {
        return [];
    }

    /** @param array<array-key, mixed> $tokenIds */
    private function validateTokenIds(array $tokenIds, Shape $shape): void
    {
        if (!array_is_list($tokenIds)) {
            throw new \InvalidArgumentException('Embedding token IDs must be a list.');
        }
        if (count($shape->dimensions) !== 2) {
            throw new \InvalidArgumentException('Embedding token shape must be rank 2 [batch, sequence].');
        }
        [$batch, $sequence] = $shape->dimensions;
        if ($batch < 0 || $sequence < 0) {
            throw new \InvalidArgumentException('Embedding batch and sequence dimensions must be non-negative.');
        }
        if ($batch !== 0 && $sequence > intdiv(PHP_INT_MAX, $batch)) {
            throw new \InvalidArgumentException('Embedding batch x sequence overflows the platform integer size.');
        }
        if (count($tokenIds) !== $batch * $sequence) {
            throw new \InvalidArgumentException('Embedding token count must equal batch x sequence.');
        }
        foreach ($tokenIds as $tokenId) {
            if (!is_int($tokenId)) {
                throw new \InvalidArgumentException('Embedding token IDs must contain only integers.');
            }
            if ($tokenId < 0 || $tokenId >= $this->vocabularySize) {
                throw new \InvalidArgumentException('Embedding token ID is outside the vocabulary.');
            }
        }
    }
}
