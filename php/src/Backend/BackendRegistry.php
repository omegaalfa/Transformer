<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Backend;

use Omegaalfa\Transformer\Exception\BackendException;

final class BackendRegistry
{
    /** @var array<string, BackendInterface> */
    private array $backends = [];

    public function register(BackendInterface $backend): void
    {
        $this->backends[$backend->type()->value] = $backend;
    }

    public function get(BackendType $type): BackendInterface
    {
        return $this->backends[$type->value]
            ?? throw new BackendException("Backend {$type->value} is not registered.");
    }
}
