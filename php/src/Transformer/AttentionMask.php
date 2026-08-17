<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Transformer;

use InvalidArgumentException;
use Omegaalfa\Transformer\Tensor\Shape;

final readonly class AttentionMask
{
    /** @var list<bool> */
    public array $values;

    /** @param array<array-key, mixed> $values */
    public function __construct(
        array $values,
        public Shape $shape,
    ) {
        if (!array_is_list($values)) {
            throw new InvalidArgumentException('Attention mask values must be a list.');
        }
        if (count($shape->dimensions) !== 2) {
            throw new InvalidArgumentException('Attention mask shape must be rank 2 [batch, sequence].');
        }
        [$batch, $sequence] = $shape->dimensions;
        if ($batch < 0 || $sequence < 0) {
            throw new InvalidArgumentException('Attention mask dimensions must be non-negative.');
        }
        if ($batch !== 0 && $sequence > intdiv(PHP_INT_MAX, $batch)) {
            throw new InvalidArgumentException('Attention mask element count overflows the platform integer size.');
        }
        if (count($values) !== $batch * $sequence) {
            throw new InvalidArgumentException('Attention mask value count must equal batch x sequence.');
        }
        foreach ($values as $value) {
            if (!is_bool($value)) {
                throw new InvalidArgumentException('Attention mask values must contain only booleans.');
            }
        }
        $this->values = $values;
    }
}
