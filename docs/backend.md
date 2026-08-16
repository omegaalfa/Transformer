# Arquitetura do Backend

`BackendInterface` compõe contratos de operações menores. `Runtime` recebe um backend através da injeção de dependência, mantendo módulos de alto nível independentes dos detalhes de execução.

- `PurePhpBackend` se tornará a referência numérica educacional.
- `FfiBackend` é o ponteiro de desenvolvimento para o tempo de execução Rust.
- `NativeExtensionBackend` se tornará o ponteiro de extensão Zend para o mesmo ABI C e tempo de execução Rust.

As operações gerais de Tensor permanecem esqueletos. `FfiBackend::addFloat32()` é uma operação de buffer explícita experimental que valida arrays PHP e delega a `NativeLibrary`:

```text
FfiBackend -> NativeLibrary -> PHP FFI -> C ABI -> Rust cdylib
```

FFI não é uma exigência permanente da API pública; alterar o ponteiro deve não exigir reescrever o tempo de execução.
