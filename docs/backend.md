# Backend architecture

`BackendInterface` composes smaller operation contracts. `Runtime` receives a
backend through dependency injection, keeping high-level modules independent of
execution details.

- `PurePhpBackend` will become the educational numerical reference.
- `FfiBackend` is the development bridge to the Rust runtime.
- `NativeExtensionBackend` will become the Zend-extension bridge to the same C
  ABI and Rust runtime.

The general Tensor operations remain skeletons. `FfiBackend::addFloat32()` is an
explicit experimental buffer operation that validates PHP arrays and delegates
to `NativeLibrary`:

```text
FfiBackend -> NativeLibrary -> PHP FFI -> C ABI -> Rust cdylib
```

FFI is not a permanent requirement of the public API; changing the bridge must
not require rewriting the runtime.
