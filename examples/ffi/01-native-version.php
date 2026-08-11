<?php

declare(strict_types=1);

use Omegaalfa\Transformer\Backend\Ffi\NativeLibrary;

require dirname(__DIR__, 2) . '/vendor/autoload.php';

$libraryPath = NativeLibrary::defaultPath(dirname(__DIR__, 2));
$library = new NativeLibrary($libraryPath);

echo 'Native runtime version: ', $library->version(), PHP_EOL;
