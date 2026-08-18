<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Backend\Ffi;

use InvalidArgumentException;
use Omegaalfa\Transformer\Backend\AbstractBackend;
use Omegaalfa\Transformer\Backend\BackendType;
use Omegaalfa\Transformer\Backend\Contract\BertBackendInterface;
use Omegaalfa\Transformer\Tensor\Shape;
use Omegaalfa\Transformer\Tensor\Tensor;
use Omegaalfa\Transformer\Transformer\AttentionMask;

final class FfiBackend extends AbstractBackend implements BertBackendInterface
{
    public function __construct(private readonly NativeLibrary $nativeLibrary)
    {
    }

    public function type(): BackendType
    {
        return BackendType::Ffi;
    }

    public function nativeVersion(): string
    {
        return $this->nativeLibrary->version();
    }

    /** @param array<array-key, mixed> $data */
    public function tensorFromFloat32(array $data, Shape $shape): Tensor
    {
        if (!array_is_list($data)) {
            throw new InvalidArgumentException('Float32 tensor data must be a list.');
        }

        return $this->nativeLibrary->tensorFromFloat32($this->normalizeFloat32List($data), $shape);
    }

    public function matmul(Tensor $left, Tensor $right): Tensor
    {
        return $left->matmul($right);
    }

    public function linear(Tensor $input, Tensor $weight, ?Tensor $bias = null): Tensor
    {
        return $input->linear($weight, $bias);
    }

    public function layerNorm(Tensor $input, Tensor $weight, Tensor $bias, float $epsilon = 1.0e-5): Tensor
    {
        return $input->layerNorm($weight, $bias, $epsilon);
    }

    /** @param array<array-key, mixed> $tokenIds */
    public function embeddingTokenIds(array $tokenIds, Shape $shape, Tensor $weight): Tensor
    {
        if (!array_is_list($tokenIds)) {
            throw new InvalidArgumentException('Embedding token IDs must be a list.');
        }
        if (count($shape->dimensions) !== 2) {
            throw new InvalidArgumentException('Embedding token shape must be rank 2 [batch, sequence].');
        }
        [$batch, $sequence] = $shape->dimensions;
        if ($batch < 0 || $sequence < 0) {
            throw new InvalidArgumentException('Embedding batch and sequence dimensions must be non-negative integers.');
        }
        $tokenCount = $this->checkedProduct($batch, $sequence, 'batch x sequence');
        if (count($tokenIds) !== $tokenCount) {
            throw new InvalidArgumentException('Embedding token count must equal batch x sequence.');
        }

        $weightShape = $weight->shape()->dimensions;
        if (count($weightShape) !== 2 || $weightShape[0] <= 0 || $weightShape[1] <= 0) {
            throw new InvalidArgumentException('Embedding weight must have shape [vocabulary_size, embedding_dim].');
        }
        $vocabularySize = $weightShape[0];
        foreach ($tokenIds as $tokenId) {
            if (!is_int($tokenId)) {
                throw new InvalidArgumentException('Embedding token IDs must contain only integers.');
            }
            if ($tokenId < 0 || $tokenId >= $vocabularySize) {
                throw new InvalidArgumentException('Embedding token ID is outside the vocabulary.');
            }
        }

        return $this->nativeLibrary->embeddingTokenIds($tokenIds, $shape, $weight);
    }

    public function add(Tensor $left, Tensor $right): Tensor
    {
        return $left->add($right);
    }

    public function softmax(Tensor $tensor, int $axis = -1): Tensor
    {
        return $tensor->softmax($axis);
    }

    public function gelu(Tensor $input): Tensor
    {
        return $this->nativeLibrary->gelu($input);
    }

    public function exactGelu(Tensor $input): Tensor
    {
        return $this->nativeLibrary->exactGelu($input);
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
        return $this->nativeLibrary->bertSelfAttention(
            $input,
            $qWeight,
            $qBias,
            $kWeight,
            $kBias,
            $vWeight,
            $vBias,
            $outWeight,
            $outBias,
            $heads,
            $mask,
        );
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
        return $this->nativeLibrary->multiHeadAttention(
            $input,
            $qWeight,
            $kWeight,
            $vWeight,
            $outWeight,
            $heads,
            $mask,
        );
    }

    public function transpose(Tensor $tensor, ?int $axisA = null, ?int $axisB = null): Tensor
    {
        return $tensor->transpose($axisA, $axisB);
    }

    /**
     * Experimental float32 buffer operation; not part of BackendInterface yet.
     *
     * @param array<array-key, mixed> $a
     * @param array<array-key, mixed> $b
     * @return list<float>
     */
    public function addFloat32(array $a, array $b): array
    {
        if (!array_is_list($a) || !array_is_list($b)) {
            throw new InvalidArgumentException('Float32 inputs must be lists.');
        }

        if (count($a) !== count($b)) {
            throw new InvalidArgumentException('Float32 inputs must have the same length.');
        }

        return $this->nativeLibrary->addFloat32(
            $this->normalizeFloat32List($a),
            $this->normalizeFloat32List($b),
        );
    }

    /**
     * Multiplies flattened row-major matrices A[M x K] and B[K x N].
     *
     * @param array<array-key, mixed> $a
     * @param array<array-key, mixed> $b
     * @return list<float>
     */
    public function matmulFloat32(array $a, array $b, int $m, int $k, int $n): array
    {
        $aLength = $this->checkedProduct($m, $k, 'M x K');
        $bLength = $this->checkedProduct($k, $n, 'K x N');
        $this->checkedProduct($m, $n, 'M x N');

        if (!array_is_list($a) || !array_is_list($b)) {
            throw new InvalidArgumentException('Float32 matrices must be flattened lists.');
        }

        if (count($a) !== $aLength || count($b) !== $bLength) {
            throw new InvalidArgumentException('Float32 matrix buffers do not match M x K and K x N.');
        }

        return $this->nativeLibrary->matmulFloat32(
            $this->normalizeFloat32List($a),
            $this->normalizeFloat32List($b),
            $m,
            $k,
            $n,
        );
    }

    /**
     * Transposes a flattened row-major matrix from rows x columns to columns x rows.
     *
     * @param array<array-key, mixed> $input
     * @return list<float>
     */
    public function transposeFloat32(array $input, int $rows, int $columns): array
    {
        $length = $this->checkedProduct($rows, $columns, 'rows x columns');

        if (!array_is_list($input)) {
            throw new InvalidArgumentException('The float32 matrix must be a flattened list.');
        }

        if (count($input) !== $length) {
            throw new InvalidArgumentException('The float32 matrix buffer does not match rows x columns.');
        }

        return $this->nativeLibrary->transposeFloat32(
            $this->normalizeFloat32List($input),
            $rows,
            $columns,
        );
    }

    /**
     * Computes a numerically stable softmax over one float32 vector.
     *
     * @param array<array-key, mixed> $input
     * @return non-empty-list<float>
     */
    public function softmaxFloat32(array $input): array
    {
        if (!array_is_list($input) || $input === []) {
            throw new InvalidArgumentException('Softmax input must be a non-empty list.');
        }

        /** @var non-empty-list<float> $normalized */
        $normalized = $this->normalizeFloat32List($input);

        foreach ($normalized as $value) {
            if (!is_finite($value)) {
                throw new InvalidArgumentException('Softmax input must contain only finite values.');
            }
        }

        return $this->nativeLibrary->softmaxFloat32($normalized);
    }

    /**
     * @param list<mixed> $values
     * @return list<float>
     */
    private function normalizeFloat32List(array $values): array
    {
        $normalized = [];

        foreach ($values as $value) {
            if (!is_int($value) && !is_float($value)) {
                throw new InvalidArgumentException('Float32 inputs must contain only numeric values.');
            }

            $normalized[] = (float) $value;
        }

        return $normalized;
    }

    private function checkedProduct(int $left, int $right, string $dimensions): int
    {
        if ($left < 0 || $right < 0) {
            throw new InvalidArgumentException("{$dimensions} dimensions cannot be negative.");
        }

        if ($left !== 0 && $right > intdiv(PHP_INT_MAX, $left)) {
            throw new InvalidArgumentException("{$dimensions} dimensions overflow the platform integer size.");
        }

        return $left * $right;
    }
}
