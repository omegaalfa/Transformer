<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Model\Loader;

use InvalidArgumentException;
use Omegaalfa\Transformer\Tensor\Shape;

final readonly class WeightMaterializationSpec
{
    public function __construct(
        public Shape $checkpointShape,
        public Shape $runtimeShape,
        public WeightOrientation $orientation = WeightOrientation::Identity,
    ) {
        $this->validateShape($checkpointShape, 'checkpoint');
        $this->validateShape($runtimeShape, 'runtime');

        $expectedRuntimeShape = match ($orientation) {
            WeightOrientation::Identity => $checkpointShape->dimensions,
            WeightOrientation::PyTorchLinearTranspose => $this->transposedShape($checkpointShape),
        };

        if ($runtimeShape->dimensions !== $expectedRuntimeShape) {
            throw new InvalidArgumentException('Runtime shape does not match the declared weight orientation.');
        }
    }

    private function validateShape(Shape $shape, string $kind): void
    {
        foreach ($shape->dimensions as $dimension) {
            if ($dimension < 0) {
                throw new InvalidArgumentException(sprintf('%s shape dimensions must be non-negative.', ucfirst($kind)));
            }
        }
    }

    /** @return array{int, int} */
    private function transposedShape(Shape $shape): array
    {
        if (count($shape->dimensions) !== 2) {
            throw new InvalidArgumentException('PyTorch Linear transposition requires a rank-2 checkpoint shape.');
        }

        return [$shape->dimensions[1], $shape->dimensions[0]];
    }
}
