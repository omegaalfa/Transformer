<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Backend;

use Omegaalfa\Transformer\Backend\Contract\ActivationBackendInterface;
use Omegaalfa\Transformer\Backend\Contract\AlgebraBackendInterface;
use Omegaalfa\Transformer\Backend\Contract\AttentionBackendInterface;
use Omegaalfa\Transformer\Backend\Contract\ElementWiseBackendInterface;
use Omegaalfa\Transformer\Backend\Contract\EmbeddingBackendInterface;
use Omegaalfa\Transformer\Backend\Contract\NormalizationBackendInterface;
use Omegaalfa\Transformer\Backend\Contract\ShapeBackendInterface;

interface BackendInterface extends AlgebraBackendInterface, ElementWiseBackendInterface, ShapeBackendInterface, ActivationBackendInterface, EmbeddingBackendInterface, NormalizationBackendInterface, AttentionBackendInterface
{
    public function type(): BackendType;
}
