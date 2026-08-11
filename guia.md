# Guia de continuidade — Native Tensor & Transformer Runtime for PHP

Este documento é a referência operacional para analisar o estado do projeto e
continuar o trabalho do ponto correto. Ele deve ser consultado antes de iniciar
qualquer milestone.

Documentos arquiteturais complementares:

- [`README.md`](README.md)
- [`docs/architecture.md`](docs/architecture.md)
- [`docs/native-runtime.md`](docs/native-runtime.md)
- [`docs/native-tensor-design.md`](docs/native-tensor-design.md)
- [`docs/roadmap.md`](docs/roadmap.md)

## Objetivo do projeto

Construir um runtime próprio para executar modelos Transformer a partir do PHP:

```text
PHP
  -> API de alto nível
  -> Backend
  -> C ABI
  -> runtime Rust
```

O primeiro produto útil não é um LLM generativo. É um encoder pequeno capaz de
produzir embeddings localmente:

```php
$embedding = $model->encode('Como solicitar férias?');
```

Esse fluxo deve funcionar sem Python, Ollama, ONNX Runtime ou chamadas HTTP em
tempo de execução.

## Arquitetura fixada

```text
PHP API
  |
Tensor / NN / Transformer
  |
BackendInterface
  |
PurePhp | FFI | futura extensão Zend
            |             |
            +--- C ABI ---+
                    |
               runtime Rust
```

Decisões aprovadas:

- Rust e Cargo são a tecnologia do runtime nativo.
- A fronteira pública é uma ABI C estável.
- PHP nunca conhece tipos ou layouts internos de Rust.
- FFI é a ponte de desenvolvimento.
- A futura extensão Zend reutilizará a mesma ABI e o mesmo runtime.
- Raw pointers existem somente em `runtime/src/ffi/`.
- Kernels recebem slices seguros, nunca raw pointers.
- Panics não podem atravessar a ABI C.
- Correção e parity vêm antes de profiling e otimização.
- APIs baseadas em buffers permanecem como referências numéricas durante a
  criação da API baseada em handles.

## Estado atual confirmado

### Infraestrutura

- [x] Rust `cdylib`.
- [x] Cargo e `Cargo.lock`.
- [x] C ABI.
- [x] PHP FFI.
- [x] `NativeLibrary`.
- [x] `FfiBackend` experimental.
- [x] Unsafe isolado na fronteira FFI.
- [x] Panic containment.
- [x] Testes Rust e integração PHP.

### Kernels de referência

- [x] `add_f32` — operação element-wise.
- [x] `matmul_f32` — álgebra matricial ingênua row-major.
- [x] `transpose_f32` — transformação estrutural de layout.
- [x] `softmax_f32` — vetor 1D numericamente estável.

O matmul ingênuo deve ser preservado como referência mesmo quando surgirem
implementações SIMD, BLAS ou GPU.

O softmax de referência usa:

```text
maximum = max(input)
exp[i]  = exp(input[i] - maximum)
output  = exp / sum(exp)
```

### Tensor milestones

- [x] **T1 — DType, Shape e Strides.**
- [x] **T2 — Storage Float32 contíguo com ownership exclusivo.**
- [x] **T3 — Tensor Rust contíguo, Float32 e CPU.**
- [x] **T4 — Handle opaco e contrato de lifecycle.**
- [x] **T5 — ABI create/destroy/shape/rank/dtype.**
- [x] **T6 — Copy-out explícito para debugging.**
- [x] **T7 — Tensor add com parity.**
- [x] **T8 — Tensor matmul com parity.**
- [x] **T9 — Tensor transpose materializado e contíguo.**
- [x] **T10 — Tensor softmax estável.**

### NN milestones

- [x] **NN-R1 — Revisão arquitetural da camada NN.**
- [ ] **NN-0 — PHP Native Tensor bridge.**
- [ ] **NN-1 — Parameter e Module.**
- [ ] **NN-2 — Linear.**
- [ ] **NN-3 — Embedding.**
- [ ] **NN-4 — LayerNorm.**
- [ ] **NN-5 — GELU.**
- [ ] **NN-R2 — Revisão antes de Attention.**

## Próximo ponto de continuação

NN-R1 foi concluída como revisão documental, sem implementar módulos ou
kernels. O próximo gate, ainda dependente de aprovação explícita, é:

```text
NN-0 — PHP Native Tensor bridge
```

NN-0 deve conectar o PHP aos handles já implementados antes de iniciar
`Parameter`, `Module` ou `Linear`. A auditoria confirmou que `NativeStorage`, o
PHP `Tensor` e as operações de handle em `NativeLibrary` ainda são skeletons.

T3 foi concluído com o seguinte contrato:

- combinar `Storage`, `Shape`, `Strides` e `DType`;
- CPU apenas;
- Float32 apenas;
- layout contíguo row-major;
- ownership exclusivo;
- validar `storage.len() == shape.numel()`;
- gerar strides contíguos a partir de Shape;
- não adicionar FFI;
- não adicionar PHP;
- não criar handles;
- não criar views ou `Arc`;
- não migrar kernels ainda.

API Rust implementada no T3:

- `Tensor::new(Storage, Shape)`;
- `Tensor::from_vec(Vec<f32>, Shape)`;
- `Tensor::zeros(Shape)`;
- leitura segura de storage, shape, strides, dtype e dados;
- transferência explícita do ownership por `into_storage()`;
- erro explícito quando `storage.len() != shape.numel()`.

T4 foi concluído com o seguinte contrato:

- `TransformerTensor` é `#[repr(C)]`, possui um `Tensor` privado e nunca expõe
  seu layout em headers C;
- C enxerga somente `typedef struct TransformerTensor TransformerTensor;`;
- a futura criação transfere um único ownership do Rust para o chamador;
- cada handle não nulo deve receber exatamente um `destroy`;
- `destroy(NULL)` será operação válida e sem efeito;
- após `destroy`, o endereço fica inválido e não pode ser reutilizado;
- double-destroy e use-after-free são violações do contrato do chamador;
- nenhuma função deixa panic Rust atravessar a C ABI;
- falhas usam status inteiro e parâmetros de saída, não apenas ponteiro nulo;
- consultas de shape copiarão dados para buffer do chamador, sem ponteiros
  emprestados para memória interna;
- a API pública de Tensor será imutável e operações futuras retornarão novos
  handles;
- a API de buffers atual permanece como referência de paridade.

T5 foi concluído com o seguinte contrato:

- `transformer_tensor_create_f32` copia os buffers de entrada e retorna o
  handle por parâmetro de saída;
- `transformer_tensor_destroy` libera ownership exclusivo e aceita `NULL`;
- rank, shape e dtype são consultados por parâmetros de saída;
- shape é copiado para buffer do chamador e valida a capacidade;
- scalar aceita `shape == NULL` com rank zero e exige um valor;
- Tensor vazio aceita `data == NULL` quando `numel == 0`;
- overflow, ponteiros nulos inválidos e capacidade insuficiente são testados;
- o header C consumido pelo PHP está sincronizado;
- não existe ainda classe PHP de Tensor nativo nem copy-out de dados.

Discriminantes públicos fixados no T5:

```text
STATUS_OK                  = 0
STATUS_INVALID_ARGUMENT    = 1
STATUS_PANIC               = 2
STATUS_INSUFFICIENT_BUFFER = 3
DTYPE_FLOAT32              = 0
```

T6 foi concluído com o seguinte contrato:

- `transformer_tensor_numel` informa quantos floats o chamador deve alocar;
- `transformer_tensor_copy_data_f32` copia somente com capacidade suficiente;
- Storage interno nunca é exposto por ponteiro emprestado;
- copy-out não consome nem invalida o handle;
- Tensor vazio aceita output nulo com capacidade zero;
- capacidade insuficiente não escreve parcialmente no buffer;
- scalar, vazio, capacidade exata, insuficiente e null possuem testes;
- o header C consumido pelo PHP está sincronizado;
- não existe ainda classe PHP de Tensor nativo.

Fluxo de materialização aprovado:

```text
transformer_tensor_numel(handle, &numel)
    -> alocar numel floats
    -> transformer_tensor_copy_data_f32(handle, output, numel)
```

T7 foi concluído com o seguinte contrato:

- `transformer_tensor_add(a, b, output)` recebe handles imutáveis;
- dtype, shape e `numel` precisam ser idênticos;
- shape diferente é rejeitado mesmo quando `numel` coincide;
- broadcasting continua proibido;
- o kernel seguro `add_f32` é reutilizado diretamente nos slices nativos;
- o resultado possui novo Storage e ownership exclusivo;
- os dois handles de entrada permanecem válidos e inalterados;
- scalar e Tensors vazios são suportados;
- a API de buffers foi preservada e usada em teste de paridade.

T8 foi concluído com o seguinte contrato:

- `transformer_tensor_matmul(a, b, output)` recebe handles imutáveis;
- A e B precisam ter rank exatamente 2 e dtype Float32;
- `[M,K] · [K,N]` retorna novo Tensor contíguo `[M,N]`;
- dimensões internas incompatíveis e ranks diferentes de 2 são rejeitados;
- o kernel Rust ingênuo permanece preservado e é reutilizado diretamente;
- os handles de entrada permanecem válidos e inalterados;
- `[2,0] · [0,2]` produz `[2,2]` preenchido com zeros;
- o resultado manual `[58,64,139,154]` possui paridade com a API de buffers;
- batch matmul, broadcasting, transposição implícita e otimizações continuam
  fora do escopo.

T9 foi concluído com o seguinte contrato:

- `transformer_tensor_transpose(input, output)` exige Tensor rank 2 Float32;
- os dados são materializados pelo kernel Rust de referência;
- `[rows, columns]` retorna novo Tensor contíguo `[columns, rows]`;
- o handle de entrada permanece válido e inalterado;
- `[2,3]` produz `[3,2]` com dados `[1,4,2,5,3,6]`;
- matrizes quadradas e dimensões zero são suportadas;
- existe paridade explícita com `transformer_transpose_f32`;
- views, troca isolada de strides e permutação arbitrária ficam adiadas.

T10 foi concluído com o seguinte contrato:

- `transformer_tensor_softmax(input, output)` exige Tensor Float32 1D não vazio;
- o resultado possui o mesmo shape e novo ownership exclusivo;
- o kernel estável subtrai o máximo antes da exponenciação;
- o handle de entrada permanece válido e inalterado;
- `[1,2,3]` possui paridade com `transformer_softmax_f32`;
- valores em torno de `1000` e `-1000` permanecem finitos e somam
  aproximadamente 1;
- valores iguais produzem distribuição uniforme;
- vazio, rank diferente de 1 e valores não finitos são rejeitados;
- axis, batch e softmax multidimensional continuam adiados.

NN-R1 foi concluída com estas decisões:

- composição, parâmetros, nomes e módulos ficam no PHP;
- storage, lifecycle e execução numérica permanecem no Rust;
- `Parameter` envolve Tensor e metadata, sem duplicar storage/shape/dtype;
- v1 é inference-only; `trainable` é metadata e não existe autograd;
- `Module` será contrato de introspecção; forwards concretos permanecem
  tipados por módulo;
- Linear precisará de projeção nativa na última dimensão com bias específico,
  não broadcasting genérico;
- Embedding aceitará IDs inteiros na fronteira de entrada, sem fingir dtype
  Float32;
- LayerNorm começará como kernel estável específico, não como coleção aleatória
  de primitivas públicas;
- GELU terá variante matemática decidida antes da implementação;
- NN-R2 é obrigatório antes de Attention.

Escopo esperado do NN-0:

- encapsular um handle opaco em exatamente um owner PHP `NativeStorage`;
- garantir destroy no máximo uma vez e impedir uso após release;
- integrar create, metadata, numel, copy-out, add, matmul, transpose e softmax
  em `NativeLibrary` sem passar por arrays entre operações;
- tornar o PHP `Tensor` capaz de representar storage nativo e metadados reais;
- manter `toArray()` como materialização explícita;
- cobrir lifecycle e cadeias de operações em testes PHP;
- não implementar `Parameter`, `Module`, Linear ou novos kernels.

O desenho completo está em [`docs/nn-design.md`](docs/nn-design.md).

## Contratos atuais do domínio Tensor

### DType

```text
DType::Float32
```

O enum está preparado para expansão, mas nenhum segundo dtype deve ser criado
antes de existir uma necessidade real.

### Shape

- Dimensões são `Vec<usize>`.
- Rank zero representa scalar.
- `Shape([]).numel() == 1`.
- Dimensões zero são válidas.
- Qualquer dimensão zero implica `numel == 0`.
- Overflow de `numel` é erro explícito.

### Strides

- Apenas strides contíguos row-major.
- `[2, 3] -> [3, 1]`.
- `[2, 3, 4] -> [12, 4, 1]`.
- Scalar possui strides vazios.
- Shapes com dimensão zero usam strides canônicos todos zero.
- Índices fora do shape e rank incompatível são erros.
- Negative e arbitrary strides não são suportados.

### Storage

- Implementação concreta em `Vec<f32>`.
- Ownership exclusivo.
- Memória contígua.
- API limitada a construção, zeros, comprimento, slices seguros e `into_vec`.
- Storage não conhece Shape, Strides, DType, device ou matemática.
- Storage não é genérico.

## Modelo de ownership planejado

Tensor v1 deverá possuir Storage diretamente:

```text
Tensor
  -> Storage
       -> Vec<f32>
```

Não usar nesta fase:

- `Arc<Storage>`;
- reference counting;
- views;
- shared mutable storage;
- borrowed native handles;
- non-contiguous tensors.

Views poderão ser avaliadas depois do T10. Inicialmente, transpose de Tensor
deverá materializar um novo Tensor contíguo.

## Modelo de handle opaco

Implementado no Rust no T4 e exposto futuramente ao C apenas como tipo
incompleto:

```c
typedef struct TransformerTensor TransformerTensor;
```

PHP guardará apenas um ponteiro opaco. Rust controlará a alocação e exatamente
uma operação de destroy deverá liberar cada Tensor não nulo.

T4 definiu status inteiro com parâmetros de saída, `destroy(NULL)` sem efeito,
metadata copiada e prefixo ABI `transformer_`. Double-free e use-after-free não
podem ser detectados de forma portável por ponteiros C crus e, portanto, são
violações documentadas do chamador. A integração com o lifecycle do objeto PHP
será tratada depois que a ABI de T5 estiver provada.

## APIs de buffers atuais

Manter durante o desenvolvimento do Tensor:

```text
transformer_tensor_add_f32
transformer_matmul_f32
transformer_transpose_f32
transformer_softmax_f32
```

Elas serão referências de parity:

```text
buffer API ≈ Tensor handle API
```

Não apagá-las ou otimizá-las prematuramente.

## Caminho até o primeiro encoder útil

Depois de T1–T10:

```text
Tensor nativo estável
  -> Linear
  -> Embedding
  -> LayerNorm / RMSNorm
  -> GELU
  -> Q / K / V
  -> scaled dot-product attention
  -> Self-Attention
  -> Multi-Head Attention
  -> Feed Forward
  -> residual connections
  -> TransformerBlock
  -> tokenizer e vocabulary
  -> config.json
  -> Safetensors
  -> encoder pequeno BERT/BGE
  -> pooling
  -> embedding real
```

Alvo inicial recomendado:

- um único modelo encoder pequeno;
- um tokenizer específico;
- CPU e Float32;
- batch 1;
- limite de sequência explícito;
- pesos Safetensors;
- mean pooling ou CLS pooling;
- comparação contra uma referência confiável.

Critérios de sucesso do primeiro produto:

- shape do embedding correta;
- valores finitos;
- norma esperada;
- parity ou similaridade dentro da tolerância definida;
- nenhuma dependência Python/ONNX/Ollama/HTTP no runtime.

## O que fica para depois

Somente após correção, parity e profiling:

- views e non-contiguous tensors;
- shared storage;
- broadcasting;
- múltiplos dtypes;
- conversão de dtype;
- SIMD;
- multithreading;
- BLAS;
- kernels fundidos;
- FP16, BF16, INT8 e INT4;
- GPU;
- decoder, causal attention, RoPE e KV Cache;
- geração, greedy, temperature, top-k e top-p.

## Protocolo obrigatório para retomar o trabalho

Ao iniciar uma nova tarefa:

1. Ler este `guia.md` por completo.
2. Ler `docs/native-tensor-design.md` e `docs/roadmap.md`.
3. Inspecionar o repositório real; não confiar apenas neste status.
4. Executar `git status` e preservar mudanças existentes.
5. Confirmar qual é o primeiro gate ainda não concluído.
6. Não iniciar o gate seguinte sem autorização explícita.
7. Não misturar dois gates na mesma implementação.
8. Manter unsafe somente em `runtime/src/ffi/`.
9. Adicionar testes pequenos com resultados calculados manualmente.
10. Executar todas as validações proporcionais ao escopo.
11. Atualizar este guia e o roadmap ao concluir o gate.

## Validações padrão

Rust:

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

PHP, quando a tarefa tocar PHP/FFI:

```bash
composer dump-autoload --optimize
vendor/bin/phpstan analyse --no-progress
vendor/bin/phpunit --colors=never
vendor/bin/php-cs-fixer fix --dry-run --diff --allow-risky=yes --using-cache=no
```

Revisão final:

```bash
git diff --check
```

Também verificar explicitamente:

- símbolos nativos exportados quando a ABI mudar;
- ausência de unsafe em kernels;
- ausência de dependências externas não autorizadas;
- ausência de Tensor/FFI/PHP quando o milestone for Rust puro;
- ausência de otimizações antes de profiling.

## Regra de parada

Ao concluir o milestone autorizado:

- validar;
- documentar;
- relatar decisões e pendências;
- parar.

Não iniciar automaticamente o próximo item do guia.
