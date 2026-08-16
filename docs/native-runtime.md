# Runtime Nativo em Rust

O runtime nativo é um crate Rust independente sob `runtime/`, construído pelo Cargo como uma `cdylib`. Ele não está implementado dentro do extensão PHP.

```text
Biblioteca PHP -> FFI -----------------+
                                      -> ABI estável de C -> Runtime Rust
Biblioteca PHP -> Extensão Zend ------+
```

O PHP nunca recebe strings, vetores, traits, panics ou tipos de propriedade.
A ABI de C usa ponteiros, comprimentos, valores primitivos e códigos de status inteiros.

## Limite de segurança

Código inseguro é confinado a `runtime/src/ffi/mod.rs`, onde ponteiros validados são convertidos em fatias emprestadas. Os kernels como
`runtime/src/kernels/add.rs` são operações Rust seguras que aceitam apenas fatias. Operações exportadas usam `catch_unwind` para garantir que um panic não possa escapar através da ABI de C.

`catch_unwind` contém panics em Rust; ele não pode tornar um ponteiro inválido seguro.
Para cada chamada de buffer não vazio, o chamador da ABI garante que:

- `a` aponta para pelo menos `length` elementos inicializados e legíveis de `f32`;
- `b` aponta para pelo menos `length` elementos inicializados e legíveis de `f32`;
- `output` aponta para pelo menos `length` elementos escritos de `f32`;
- todos os três buffers permanecem válidos durante a chamada completa;
- `output` não sobrepõe os buffers de entrada.

O limite pode rejeitar ponteiros nulos e comprimentos impossíveis, mas um ponteiro não nulo não prova que a memória é válida. Violar o contrato do chamador pode causar comportamento indefinido antes que Rust possa retornar um código de status.

A ABI de adição retorna:

```text
0 = OK
1 = ARGUMENTO INVÁLIDO
2 = PANIC contido no limite da ABI
```

## Decisão sobre Tensor Opaque

```text
Tensor PHP
    |
    +-- handle / ponteiro nativo
              |
              v
         Tensor Nativo
              +-- buffer
              +-- shape
              +-- strides
              +-- dtype
              +-- device
```

Buffers nativos permanecem nativos entre operações. `NativeStorage` possui o handle opaco do lado PHP, e a conversão para um array PHP é explícita através de `Tensor::toFloat32()`. A propriedade Rust governa as vidas nativas; a ABI de C nunca expõe tipos Rust.

## Modelos atuais de execução

Os APIs legados de buffer intencionalmente realizam cópias:

```text
array PHP -> buffer FFI -> kernel Rust -> buffer FFI -> array PHP
```

Eles permanecem disponíveis para compatibilidade e paridade numérica.

O caminho orientado à produção do PHP usa handles residentes:

```text
array PHP
    -> tensorFromFloat32()       uma cópia de entrada
    -> Tensor Nativo
    -> matmul/add/softmax/...    sem materialização no PHP
    -> Tensor Nativo
    -> toFloat32()               cópia explícita de saída
```

Cada operação retorna um novo handle proprietário; os inputs permanecem vivos e inalterados.
`destroy()` libera um handle cedo, enquanto o destrutor PHP é a fallback.

A progressão implementada é:

```text
native_version -> add_f32 -> matmul_f32 -> transpose_f32 -> softmax_f32
    -> operações básicas estáveis
    -> Tensor Nativo / Shape / Strides / DType / handles opacos
    -> bridge PHP Tensor / NativeStorage residente
```

Futuros kernels incluem normalização, atenção, ativação, cache e quantização. SIMD, BLAS, threads, GPU e matmul otimizado seguem a corretude, paridade e perfilamento. O matmul naive permanece disponível como referência numérica em vez de ser substituído por implementações otimizadas.

O 1D softmax de referência permanece disponível e está numericamente estável. A operação aditiva `transformer_tensor_softmax_last_dim` suporta tensores contínuos de rank-N
e normaliza cada linha ao longo do último eixo em uma única chamada nativa.
