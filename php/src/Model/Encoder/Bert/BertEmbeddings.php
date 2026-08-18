<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Model\Encoder\Bert;

use InvalidArgumentException;
use Omegaalfa\Transformer\NN\Embedding;
use Omegaalfa\Transformer\NN\LayerNorm;
use Omegaalfa\Transformer\NN\Module;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Tensor\Shape;
use Omegaalfa\Transformer\Tensor\Tensor;

final readonly class BertEmbeddings implements Module
{
    public function __construct(
        public BertConfig $config,
        public Runtime $runtime,
        public Embedding $word,
        public Embedding $position,
        public Embedding $tokenType,
        public LayerNorm $norm,
    ) {
        if ($word->vocabularySize !== $config->vocabularySize || $word->dimensions !== $config->hiddenSize
            || $position->vocabularySize !== $config->maxPositionEmbeddings || $position->dimensions !== $config->hiddenSize
            || $tokenType->vocabularySize !== $config->typeVocabularySize || $tokenType->dimensions !== $config->hiddenSize
            || $norm->normalizedSize !== $config->hiddenSize
            || $word->runtime !== $runtime || $position->runtime !== $runtime
            || $tokenType->runtime !== $runtime || $norm->runtime !== $runtime) {
            throw new InvalidArgumentException('BERT embedding modules must match config and runtime.');
        }
    }

    /**
     * @param array<array-key, mixed>      $inputIds
     * @param array<array-key, mixed>|null $tokenTypeIds
     */
    public function forward(array $inputIds, Shape $shape, ?array $tokenTypeIds = null): Tensor
    {
        if (count($shape->dimensions) !== 2) {
            throw new InvalidArgumentException('BERT input shape must be [batch, sequence].');
        }
        [$batch, $sequence] = $shape->dimensions;
        if ($batch < 0 || $sequence < 0 || $sequence > $this->config->maxPositionEmbeddings) {
            throw new InvalidArgumentException('BERT batch and sequence must be non-negative and sequence within maximum positions.');
        }
        $count = $this->checkedProduct($batch, $sequence);
        if (!array_is_list($inputIds) || count($inputIds) !== $count) {
            throw new InvalidArgumentException('BERT input IDs must be a list matching batch x sequence.');
        }
        $validatedInputIds = $this->integerIds($inputIds, $this->config->vocabularySize, 'input');
        $tokenTypeIds ??= array_fill(0, $count, 0);
        if (!array_is_list($tokenTypeIds) || count($tokenTypeIds) !== $count) {
            throw new InvalidArgumentException('BERT token type IDs must be a list matching batch x sequence.');
        }
        $validatedTokenTypeIds = $this->integerIds($tokenTypeIds, $this->config->typeVocabularySize, 'token type');
        $positionIds = [];
        for ($row = 0; $row < $batch; ++$row) {
            for ($position = 0; $position < $sequence; ++$position) {
                $positionIds[] = $position;
            }
        }
        $word = $this->word->forwardTokenIds($validatedInputIds, $shape);
        $position = $this->position->forwardTokenIds($positionIds, $shape);
        $tokenType = $this->tokenType->forwardTokenIds($validatedTokenTypeIds, $shape);

        return $this->norm->forward($word->add($position)->add($tokenType));
    }

    public function parameters(): array
    {
        return [];
    }

    public function modules(): array
    {
        return ['word' => $this->word, 'position' => $this->position, 'token_type' => $this->tokenType, 'norm' => $this->norm];
    }

    private function checkedProduct(int $batch, int $sequence): int
    {
        if ($batch !== 0 && $sequence > intdiv(PHP_INT_MAX, $batch)) {
            throw new InvalidArgumentException('BERT batch x sequence overflows the platform integer size.');
        }

        return $batch * $sequence;
    }

    /**
     * @param list<mixed> $ids
     * @return list<int>
     */
    private function integerIds(array $ids, int $upperBound, string $kind): array
    {
        $validated = [];
        foreach ($ids as $id) {
            if (!is_int($id) || $id < 0 || $id >= $upperBound) {
                throw new InvalidArgumentException("BERT {$kind} ID is outside the configured vocabulary.");
            }
            $validated[] = $id;
        }
        return $validated;
    }
}
