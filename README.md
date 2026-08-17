<div align="center">

# Transformer para PHP

**Runtime nativo de Tensors e Transformers, escrito em Rust e projetado para PHP.**

PHP para composição e ergonomia. Rust para memória, álgebra e execução numérica.

![PHP 8.4+](https://img.shields.io/badge/PHP-8.4%2B-777BB4?logo=php&logoColor=white)
![Rust 2021](https://img.shields.io/badge/Rust-2021-000000?logo=rust&logoColor=white)
![FFI](https://img.shields.io/badge/integração-FFI-4F5D95)
![Licença MIT](https://img.shields.io/badge/licença-MIT-22c55e)

</div>

---

## Visão geral

Este projeto está construindo um runtime de Transformer que pode ser consumido
diretamente pelo PHP, sem depender de Python, ONNX ou serviços externos para a
execução numérica.

O PHP será responsável pela API de alto nível, composição dos modelos,
tokenização e carregamento de configurações. O Rust será responsável pelos
Tensors nativos, ownership de memória, validação e operações matemáticas.

```text
PHP — API, modelos e composição
 │
 ▼
C ABI — contrato estável
 │
 ▼
Rust — Tensor, memória e execução numérica
```

O objetivo inicial não é reproduzir um LLM completo. O primeiro produto útil é
um encoder capaz de gerar embeddings nativamente:

```php
$embedding = $model->encode('Como solicitar férias?');
```

## Estado atual

A fase fundamental do Tensor nativo, dividida nos gates T1–T10, está concluída.

| Área | Estado |
|---|:---:|
| Rust `cdylib` e Cargo | ✅ |
| C ABI e integração PHP FFI | ✅ |
| `DType`, `Shape`, `Strides` e `Storage` | ✅ |
| Tensor nativo Float32/CPU | ✅ |
| Handle opaco e lifecycle | ✅ |
| Metadata e copy-out explícito | ✅ |
| Add, matmul, transpose e softmax | ✅ |
| Paridade com as APIs de buffers | ✅ |
| Bridge de Tensor nativo no PHP | ✅ |
| Camada NN | ⬜ |
| Attention e Transformer | ⬜ |

As operações entre handles permanecem inteiramente no Rust:

```text
Tensor* A ─┐
           ├─► operação Rust ─► novo Tensor*
Tensor* B ─┘
```

Isso permite encadear diretamente pela API PHP:

```text
Tensor
  └─► matmul
        └─► transpose
              └─► softmax
                    └─► Tensor
```

`Tensor` mantém um `NativeStorage` proprietário. Cada operação retorna outro
Tensor residente, sem converter resultados intermediários em arrays PHP. A
conversão acontece somente quando o usuário chama `toFloat32()`.

## Operações nativas disponíveis

### APIs de referência baseadas em buffers

- `transformer_tensor_add_f32`
- `transformer_matmul_f32`
- `transformer_transpose_f32`
- `transformer_softmax_f32`

### APIs baseadas em handles

- criação e destruição de Tensor;
- consultas de rank, shape, dtype e `numel`;
- copy-out Float32 com capacidade validada;
- `transformer_tensor_add`;
- `transformer_tensor_matmul`;
- `transformer_tensor_transpose`;
- `transformer_tensor_softmax` para vetores;
- `transformer_tensor_softmax_last_dim` para Tensor rank-N.

As APIs de buffers continuam preservadas como referência numérica para testes
de paridade. Elas não serão removidas até que a camada baseada em handles esteja
completamente consolidada.

## Arquitetura

```text
┌──────────────────────────────────────────┐
│ PHP                                      │
│ Tensor · NN · Transformer · Model        │
└───────────────────┬──────────────────────┘
                    │
           Backend / NativeLibrary
                    │
┌───────────────────▼──────────────────────┐
│ C ABI                                    │
│ ponteiros opacos · status · lifecycle    │
└───────────────────┬──────────────────────┘
                    │
┌───────────────────▼──────────────────────┐
│ Rust Runtime                             │
│ Tensor · Storage · Shape · Kernels       │
└──────────────────────────────────────────┘
```

Ponteiros crus existem somente na borda FFI. Os kernels recebem slices seguros
e não conhecem PHP, CData ou detalhes do lifecycle externo:

```text
raw pointers
    → validação na FFI
    → &[f32] / &mut [f32]
    → kernel Rust seguro
```

Documentação arquitetural:

- [Arquitetura geral](docs/architecture.md)
- [Uso do Tensor no PHP](docs/tensor.md)
- [Backends](docs/backend.md)
- [Runtime nativo](docs/native-runtime.md)
- [Design do Tensor nativo](docs/native-tensor-design.md)
- [Revisão da camada NN](docs/nn-design.md)
- [Roadmap](docs/roadmap.md)
- [Guia de continuidade](guia.md)

## Requisitos

- PHP 8.4 ou superior com FFI habilitada;
- Rust e Cargo compatíveis com a edição 2021;
- Composer para instalar e executar as ferramentas de desenvolvimento.

Verifique o ambiente:

```bash
php --version
php -r 'var_dump(extension_loaded("FFI"));'
rustc --version
cargo --version
composer --version
```

## Instalação

```bash
composer install
```

Compile o runtime nativo em modo release:

```bash
cargo build --manifest-path runtime/Cargo.toml --release
```

Artefatos esperados:

| Plataforma | Biblioteca |
|---|---|
| Linux/WSL | `runtime/target/release/libtransformer_runtime.so` |
| macOS | `runtime/target/release/libtransformer_runtime.dylib` |
| Windows | `runtime/target/release/transformer_runtime.dll` |

> FFI não compila o Rust. Ela carrega a biblioteca compartilhada previamente
> produzida pelo Cargo.

## Exemplos

Depois do build release:

Os exemplos `02`–`05` mantêm as APIs de array para demonstrar compatibilidade;
o exemplo `06` mostra o fluxo residente recomendado para pipelines.

```bash
php examples/ffi/01-native-version.php
php examples/ffi/02-tensor-add.php
php examples/ffi/03-matmul.php
php examples/ffi/04-transpose.php
php examples/ffi/05-softmax.php
php examples/ffi/06-native-tensor-pipeline.php
```

### Início rápido: Tensor residente

```php
<?php

use Omegaalfa\Transformer\Backend\Ffi\FfiBackend;
use Omegaalfa\Transformer\Backend\Ffi\NativeLibrary;
use Omegaalfa\Transformer\Tensor\Shape;

require __DIR__ . '/vendor/autoload.php';

$backend = new FfiBackend(new NativeLibrary(
    NativeLibrary::defaultPath(__DIR__),
));

$input = $backend->tensorFromFloat32(
    [1, 2, 3, 4, 5, 6],
    new Shape([2, 3]),
);

// Uma chamada nativa: cada linha é normalizada no último eixo.
$output = $input->softmax();

print_r(array_chunk($output->toFloat32(), 3));

// Opcional: libera imediatamente. O destrutor também funciona como fallback.
$output->destroy();
$input->destroy();
```

Execute a partir da raiz do projeto, depois do build release:

```bash
php examples/ffi/05-softmax.php
```

Saída aproximada para as duas linhas:

```text
[
  [0.09003057, 0.24472848, 0.66524094],
  [0.09003057, 0.24472848, 0.66524094]
]
```

`tensorFromFloat32()` faz a cópia inicial do array PHP. `matmul()`, `add()`,
`softmax()` e `transpose()` mantêm os dados no runtime. `toFloat32()` é a
fronteira explícita que copia o resultado de volta para PHP.

Resultados de referência:

```text
[1, 2, 3] + [10, 20, 30]
= [11, 22, 33]
```

```text
[1 2 3]   [ 7  8]   [ 58  64]
[4 5 6] × [ 9 10] = [139 154]
            [11 12]
```

```text
transpose([1 2 3])   [1 4]
          [4 5 6]  = [2 5]
                     [3 6]
```

```text
softmax([1, 2, 3])
≈ [0.09003057, 0.24472848, 0.66524094]
```

O softmax subtrai o maior valor antes da exponenciação, permanecendo estável
para entradas como `[1000, 1001, 1002]`.

## Qualidade e testes

Execute toda a validação PHP:

```bash
composer check
```

Execute as validações do runtime:

```bash
cargo fmt --manifest-path runtime/Cargo.toml --check
cargo test --manifest-path runtime/Cargo.toml
cargo clippy \
    --manifest-path runtime/Cargo.toml \
    --all-targets \
    --all-features \
    -- -D warnings
cargo build --manifest-path runtime/Cargo.toml --release
```

Estado validado após a bridge PHP e o softmax no último eixo:

- **107 testes Rust**;
- **28 testes PHP / 98.373 asserções**;
- Clippy sem warnings;
- PHPStan sem erros;
- PHP CS Fixer limpo;
- build release concluído.

## Roadmap

```text
FASE TENSOR
  T1–T10                                      ✅ concluída

FASE NN
  NN-R1  revisão arquitetural                 ✅ concluída
  NN-0   bridge PHP para Tensor nativo        ✅ concluída
  NN-1   Parameter + Module                   ✅ concluída
  NN-2   Linear                               ✅ concluída
  NN-3   Embedding                            ✅ concluída
  NN-4   LayerNorm                            ✅ concluída
  NN-5   GELU                                 ⬜
  NN-R2  revisão antes de Attention           ⬜

FASE TRANSFORMER
  Q / K / V                                   ⬜
  Self-Attention                              ⬜
  Multi-Head Attention                        ⬜
  Feed Forward                                ⬜
  Residual + TransformerBlock                 ⬜

MODELO REAL
  tokenizer e vocabulário                     ⬜
  config e Safetensors                        ⬜
  encoder BERT/BGE                            ⬜
  pooling e embeddings                        ⬜
```

A revisão NN-R1 determinou que não serão adicionadas operações numéricas
aleatórias. Cada novo kernel deverá existir para atender uma necessidade
concreta de `Linear`, `Embedding`, `LayerNorm`, `GELU` ou Attention.

## Extensão Zend

O fluxo atual utiliza PHP FFI. Uma extensão Zend poderá consumir a mesma C ABI
e o mesmo runtime Rust no futuro:

```text
PHP → Zend Extension → C ABI → Rust Runtime
```

Não é necessário usar `phpize`, `configure` ou editar o `php.ini` nesta fase.

## Solução de problemas

- Confirme que FFI está habilitada com
  `php -r 'var_dump(extension_loaded("FFI"));'`.
- Verifique se o perfil compilado é o mesmo carregado pelo PHP.
- Após alterações no Rust, execute novamente o build correspondente.
- No Linux/WSL, inspecione símbolos com
  `nm -D runtime/target/release/libtransformer_runtime.so`.
- Se a biblioteca não for encontrada, confira o caminho calculado por
  `NativeLibrary::defaultPath()`.

## LayerNorm nativo (NN-4)

O módulo `LayerNorm` executa normalização de inferência sobre a última
dimensão por meio do backend FFI, com gamma/beta residentes e `epsilon=1e-5`
por default. O kernel usa Welford/Float64, produz um Tensor independente e
rejeita shapes, epsilon ou valores não finitos antes de publicar o output.

## Licença

Distribuído sob a licença [MIT](LICENSE).
