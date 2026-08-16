# Arquitetura do Backend

`BackendInterface` compõe contratos de operações menores. `Runtime` recebe um backend através da injeção de dependência, mantendo módulos de alto nível independentes dos detalhes de execução.

- `PurePhpBackend` se tornará a referência numérica educacional.
- `FfiBackend` é o ponteiro de desenvolvimento para o tempo de execução Rust.
- `NativeExtensionBackend` se tornará o ponteiro de extensão Zend para o mesmo ABI C e tempo de execução Rust.

`FfiBackend` cria Tensors residentes com `tensorFromFloat32()` e implementa
`add`, `matmul`, `transpose` e `softmax` sobre handles nativos:

```php
$tensor = $backend->tensorFromFloat32($values, new Shape([128, 768]));
$result = $backend->softmax($tensor); // último eixo, sem materialização
$values = $result->toFloat32();       // copy-out explícito
```

Fluxo residente:

```text
Tensor PHP -> NativeStorage -> handle opaco -> kernel Rust -> novo handle
```

As operações `addFloat32()`, `matmulFloat32()`, `transposeFloat32()` e
`softmaxFloat32()` continuam disponíveis para compatibilidade e paridade. Elas
recebem e retornam arrays, portanto fazem cópias na fronteira FFI:

```text
FfiBackend -> NativeLibrary -> PHP FFI -> C ABI -> Rust cdylib
```

FFI não é uma exigência permanente da API pública; trocar a bridge por uma
extensão Zend não deve exigir reescrever os kernels ou a abstração de Tensor.
