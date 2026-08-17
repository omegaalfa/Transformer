<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\NN;

interface Module
{
    /** @return array<string, Parameter> */
    public function parameters(): array;

    /** @return array<string, Module> */
    public function modules(): array;
}
