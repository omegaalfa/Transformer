<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tests\Unit\Transformer;

use InvalidArgumentException;
use Omegaalfa\Transformer\Tensor\Shape;
use Omegaalfa\Transformer\Transformer\AttentionMask;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;

final class AttentionMaskTest extends TestCase
{
    public function testStoresAFlatBooleanRowMajorMask(): void
    {
        $mask = new AttentionMask([true, false, false, true], new Shape([2, 2]));

        self::assertSame([true, false, false, true], $mask->values);
        self::assertSame([2, 2], $mask->shape->dimensions);
    }

    /** @return iterable<string, array{list<bool>, list<int>}> */
    public static function emptyShapes(): iterable
    {
        yield 'zero batch' => [[], [0, 3]];
        yield 'zero sequence' => [[], [2, 0]];
        yield 'both zero' => [[], [0, 0]];
    }

    /**
     * @param list<bool> $values
     * @param list<int> $shape
     */
    #[DataProvider('emptyShapes')]
    public function testAcceptsApprovedEmptyShapes(array $values, array $shape): void
    {
        self::assertSame($shape, (new AttentionMask($values, new Shape($shape)))->shape->dimensions);
    }

    /** @return iterable<string, array{array<array-key, mixed>, list<int>}> */
    public static function invalidMasks(): iterable
    {
        yield 'not a list' => [[1 => true], [1, 1]];
        yield 'rank one' => [[true], [1]];
        yield 'rank three' => [[true], [1, 1, 1]];
        yield 'negative batch' => [[], [-1, 0]];
        yield 'wrong count' => [[true], [1, 2]];
        yield 'integer value' => [[1], [1, 1]];
        yield 'string value' => [['1'], [1, 1]];
    }

    /**
     * @param array<array-key, mixed> $values
     * @param list<int> $shape
     */
    #[DataProvider('invalidMasks')]
    public function testRejectsMalformedMasks(array $values, array $shape): void
    {
        $this->expectException(InvalidArgumentException::class);
        new AttentionMask($values, new Shape($shape));
    }
}
