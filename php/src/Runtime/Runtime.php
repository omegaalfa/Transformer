<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Runtime;

use Omegaalfa\Transformer\Backend\BackendInterface;

final readonly class Runtime
{
    public function __construct(
        private BackendInterface $backend,
        public RuntimeConfig $config,
    ) {
    }

    public function backend(): BackendInterface
    {
        return $this->backend;
    }
}
