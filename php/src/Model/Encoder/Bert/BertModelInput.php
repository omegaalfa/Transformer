<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Model\Encoder\Bert;

use Omegaalfa\Transformer\Tensor\Shape;
use Omegaalfa\Transformer\Transformer\AttentionMask;

final readonly class BertModelInput
{
    /**
     * @param list<int>      $inputIds
     * @param list<int>|null $tokenTypeIds
     */
    public function __construct(
        public array $inputIds,
        public Shape $shape,
        public ?AttentionMask $attentionMask = null,
        public ?array $tokenTypeIds = null,
    ) {
    }
}
