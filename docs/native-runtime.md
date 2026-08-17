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

## Lifecycle residente para inferência repetida

Pesos imutáveis devem ser convertidos para Tensor nativo uma vez durante o
setup da instância que executará a inferência. Enquanto shape, dtype e conteúdo
não mudarem, o mesmo Tensor pode participar de qualquer número de operações:

```php
$input = $backend->tensorFromFloat32($inputData, $inputShape);
$weights = $backend->tensorFromFloat32($weightData, $weightShape);
$residual = $backend->tensorFromFloat32($residualData, $residualShape);

for ($iteration = 0; $iteration < $repetitions; ++$iteration) {
    $projected = $input->matmul($weights);
    $added = $projected->add($residual);
    $normalized = $added->softmax();
    $output = $normalized->transpose();

    $projected->destroy();
    $added->destroy();
    $normalized->destroy();
    // consumir ou destruir $output antes da próxima iteração
}

$input->destroy();
$weights->destroy();
$residual->destroy();
```

O objeto que representa o modelo ou serviço de inferência deve possuir os
pesos. Ele os cria no setup, empresta-os imutavelmente durante `forward()` e os
libera no teardown. Não existe registry global, singleton ou cache implícito.
Destruir um conjunto não afeta outro conjunto criado pela mesma `NativeLibrary`.

Inputs são diferentes de pesos. Se os dados de entrada mudam entre inferências,
crie um novo Tensor de input para cada valor e mantenha apenas os pesos
residentes. A API atual não oferece atualização in-place; não reutilize um input
antigo para dados diferentes. O residual só deve ser residente quando for
realmente constante para todas as chamadas.

Uma falha de operação não consome os inputs. Outputs criados antes de uma
exceção são liberados pelos destrutores PHP, e A/B/residual continuam válidos.
Chamadas sequenciais na mesma instância e instâncias independentes são
suportadas. Execução PHP concorrente sobre o mesmo objeto não possui contrato
adicional nesta etapa; nenhum mutex ou paralelismo foi introduzido.

Veja `examples/ffi/06-native-tensor-pipeline.php` para um lifecycle completo e
executável. `TRANSFORMER_EXAMPLE_REPETITIONS` controla o número de inferências.

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

## Linear na última dimensão

O símbolo aditivo `transformer_tensor_linear_last_dim` recebe input, weight e
bias opcional como handles imutáveis. Input pode ter qualquer rank maior ou
igual a um; weight é `[input_features, output_features]` e bias, quando presente,
é `[output_features]`. O output preserva todas as dimensões anteriores e troca
somente a última por `output_features`.

No CPU, a operação reutiliza `matmul_dispatch_f32` com as dimensões anteriores
achatadas conceitualmente em M e aplica bias diretamente por linha. O bias não
é expandido nem copiado para o shape completo. Todos os inputs permanecem vivos
e o output recebe um novo handle proprietário.

## Embedding com IDs inteiros

O símbolo aditivo `transformer_tensor_embedding` recebe um buffer temporário
`int64_t` row-major para `[B,S]`, um weight residente Float32 `[V,D]` e publica
um novo handle `[B,S,D]`. IDs são validados como `0 <= id < V` antes da cópia e
só então convertidos de `i64` para `usize` no Rust.

O kernel recebe exclusivamente slices seguros e copia
`output[b,s,:] = weight[token_ids[b*S+s],:]`. Não existe Tensor inteiro,
gather PHP ou Tensor intermediário. O weight permanece imutável e residente;
o CData de IDs vive somente durante a chamada. `[0,S]`, `[B,0]` e `[0,0]`
produzem outputs vazios com o shape completo e não acessam nenhuma linha.

`Module` cobre introspecção e `TensorModule` cobre operações Tensor-to-Tensor.
Por isso `Embedding` usa `forwardTokenIds()` sem representar IDs como Tensor
Float32. O backend PurePHP declara o contrato, mas mantém o erro explícito de
operação não implementada nesta etapa.

## LayerNorm Tensor ABI

`transformer_tensor_layer_norm(input, weight, bias, epsilon, output)` é uma
extensão aditiva da ABI. Os três handles são emprestados e imutáveis; o output
é um novo handle publicado apenas após validação e execução completas. A
operação exige input rank-N com `N>=1`, última dimensão `D>0`, weight/bias
rank-1 `[D]`, Float32 contíguo row-major e valores finitos. Panics são contidos
na fronteira C. Shapes como `[0,D]`, `[B,0,D]` e `[0,0,D]` retornam Tensors
vazios do mesmo shape sem executar Welford.

## GELU Tensor ABI

O símbolo aditivo `transformer_tensor_gelu(input, output)` aplica a aproximação
tanh GELU elemento a elemento. O input é emprestado e imutável; o output é um
novo Tensor Float32 contíguo, row-major e com shape idêntico. O cálculo usa
temporários Float64, rejeita NaN/Inf e só publica o handle após validar toda a
execução. Rank zero e qualquer shape vazio são válidos. O módulo PHP é
stateless, não possui Parameters e rejeita inputs de outra `NativeLibrary`.

## MultiHeadAttention Tensor ABI

O símbolo aditivo `transformer_tensor_multi_head_attention` recebe input
`[B,S,D]`, quatro weights residentes `[D,D]`, `H` e uma máscara booleana
opcional. Ausência de máscara é `NULL + 0`; uma máscara presente atravessa a
FFI como `uint8_t[B*S]`, aceitando somente 0/1. O buffer é temporário e continua
fora do sistema de dtype do Tensor.

O runtime valida rank 3, shapes, Float32 contíguo row-major, `D>0`, `H>0`,
divisibilidade, overflow, máscara e finitude. Os heads usam layout lógico
`[B,H,S,Dh]`; scores e probabilidades usam `[B,H,S,S]`. Uma máscara false
exclui a key do softmax e um batch totalmente mascarado falha. Shapes com
`B=0` ou `S=0` retornam Tensor vazio sem executar kernels numéricos.

Todos os handles de entrada são imutáveis e emprestados. O output começa nulo,
é publicado somente depois da execução completa e possui Storage independente.
Panics retornam status de panic e não atravessam a ABI. A operação permanece
CPU/Float32 e não se conecta ao Graph Executor experimental.

## FeedForward e TransformerBlock sem ABI própria

NN-7 não adiciona símbolos nativos. `FeedForward` compõe as ABIs existentes de
Linear e GELU, enquanto `TransformerBlock` compõe LayerNorm,
MultiHeadAttention, `Tensor::add()` e FeedForward em PHP. Os Tensors
intermediários continuam residentes no runtime nativo e são liberados pelo
lifecycle normal de `NativeStorage`; nenhuma lista PHP é materializada durante
o forward. Não existem kernels fused ou nós novos no Graph Executor.
