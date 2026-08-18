<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tests\Unit\Backend;

use Omegaalfa\Transformer\Backend\Cuda\CudaBgePrecision;
use PHPUnit\Framework\TestCase;

final class CudaBgePrecisionTest extends TestCase
{
    public function testPrecisionModesHaveStableAbiValues(): void
    {
        self::assertSame(0, CudaBgePrecision::Float32->value);
        self::assertSame(1, CudaBgePrecision::Float16->value);
        self::assertSame(2, CudaBgePrecision::BFloat16->value);
    }
}
