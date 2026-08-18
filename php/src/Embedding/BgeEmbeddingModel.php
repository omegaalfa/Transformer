<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Embedding;

use InvalidArgumentException;
use Omegaalfa\Transformer\Embedding\Normalization\L2Normalizer;
use Omegaalfa\Transformer\Embedding\Pooling\ClsPooling;
use Omegaalfa\Transformer\Embedding\Pooling\MeanPooling;
use Omegaalfa\Transformer\Embedding\Pooling\PoolingInterface;
use Omegaalfa\Transformer\Model\Encoder\Bert\BertModelInput;
use Omegaalfa\Transformer\Model\Encoder\Bert\BertModelInterface;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Tensor\Tensor;
use Omegaalfa\Transformer\Tokenizer\BertTokenizer;

final readonly class BgeEmbeddingModel implements EmbeddingModelInterface
{
    private PoolingInterface $pooling;
    private L2Normalizer $normalizer;

    public function __construct(
        public BertModelInterface $model,
        public BertTokenizer $tokenizer,
        public Runtime $runtime,
        public BgePoolingStrategy $poolingStrategy = BgePoolingStrategy::Cls,
    ) {
        $this->pooling = match ($poolingStrategy) {
            BgePoolingStrategy::Cls => new ClsPooling($runtime),
            BgePoolingStrategy::Mean => new MeanPooling($runtime),
        };
        $this->normalizer = new L2Normalizer($runtime);
    }

    /** @return list<float> */
    public function encode(string $text): array
    {
        return $this->encodeBatch([$text])[0];
    }

    /** @param list<string> $texts
     *  @return list<list<float>>
     */
    public function encodeBatch(array $texts): array
    {
        $output = $this->encodeBatchOutput($texts);
        $width = $output->embedding->shape()->dimensions[1];
        if ($width < 1) {
            throw new InvalidArgumentException('BGE embedding width must be positive.');
        }

        return array_chunk($output->embedding->toFloat32(), $width);
    }

    public function encodeTensor(string $text): Tensor
    {
        return $this->encodeBatchOutput([$text])->embedding;
    }

    /** @param list<string> $texts */
    public function encodeBatchOutput(array $texts): BgeEmbeddingOutput
    {
        $tokens = $this->tokenizer->encodeBatch($texts);
        $hidden = $this->model->forward(new BertModelInput(
            $tokens->inputIds,
            $tokens->shape,
            $tokens->attentionMask,
            $tokens->tokenTypeIds,
        ))->lastHiddenState;
        $maskValues = array_map(static fn (bool $value): float => $value ? 1.0 : 0.0, $tokens->attentionMask->values);
        $mask = $this->runtime->backend()->tensorFromFloat32($maskValues, $tokens->shape);
        $pooled = $this->pooling->pool($hidden, $mask);
        $embedding = $this->normalizer->normalize($pooled);
        $shape = $embedding->shape()->dimensions;
        $config = $this->model->modelConfig();
        if ($shape !== [count($texts), $config->transformer->hiddenSize]) {
            throw new InvalidArgumentException('BGE embedding output shape does not match the model configuration.');
        }

        return new BgeEmbeddingOutput($pooled, $embedding);
    }
}
