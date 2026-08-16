# Design de Tensor Nativo

Status: **T1 até T10 implementados; NN-R1 completo; o próximo passo é a ponte do PHP handle**.

As APIs de buffer validadas estabeleceram o C ABI, PHP FFI, limite seguro da kernel, indexação em linha única, validação dimensional e comportamento numérico. Eles não são o modelo de execução final porque cada chamada atualmente copia dados entre arrays PHP e buffers nativos.

## Objetivo

Manter os dados do tensor em Rust ao longo das cadeias de operações:

```text
Tensor PHP
    |
    +-- handle opaco
            |
            v
       Tensor Rust
            +-- adição
            +-- multiplicação matricial
            +-- transposição
            +-- softmax
```

Apenas uma operação explícita de depuração/materialização copia dados de volta para PHP.

## Limites do Tensor v1

O primeiro tensor nativo é deliberadamente conservador:

- CPU apenas;
- Float32 apenas;
- armazenamento contínuo em linha única apenas;
- o tensor exclusivamente possui seu armazenamento;
- sem visualizações ou armazenamento compartilhado;
- sem broadcasting;
- sem conversão implícita de tipo de dados;
- sem operações públicas in-place;
- as operações retornam novos tensores proprietários.

Essas restrições reduzem a simultaneidade de incerteza enquanto deixam claros pontos de extensão para futuras marcos.

## Responsabilidades

Conceptualmente, o modelo é:

```rust
pub struct Tensor {
    storage: Storage,
    shape: Shape,
    strides: Strides,
    dtype: DType,
}
```

Isso é a estrutura T3 confirmada em `runtime/src/tensor/tensor.rs`.

### DType

Descreve como os elementos de armazenamento são interpretados. Tensor v1 suporta apenas
`Float32`, mas o tipo de dados permanece explícito para que o tensor não se torne permanentemente
acoplado a uma representação específica.

### Shape

Detém uma lista ordenada de tamanhos de dimensão. Ele é responsável pela verificação calculada do número de elementos e rejeita produtos de dimensões que estouram `usize`.

### Strides

Mapeia as coordenadas lógicas para deslocamentos de armazenamento. Os passos contínuos em linha única são derivados da forma usando multiplicação verificada.

```text
forma   = [2, 3]
passos = [3, 1]

índice [1, 2]
deslocamento = 1*3 + 2*1 = 5
```

Tensor v1 armazena explicitamente passos contínuos, mesmo que possam ser derivados. Isso permite que invariants e semânticas de visualização futuras evoluam sem alterar a fronteira de responsabilidade do tensor.

Para qualquer forma com tamanho zero, T1 usa passos canônicos todos-zeros. Nenhum índice lógico é válido para tal forma, então esses passos nunca são usados para acessar o armazenamento; a convenção também evita estouro de produto de deslocamento irrelevante em layouts vazios.

### Armazenamento

Detém a alocação nativa como um `Vec<f32>` concreto com propriedade exclusiva. Sua API é intencionalmente limitada à construção, alocação zero, comprimento, acesso seguro ao fatiamento e transferência de volta em um Vec. O armazenamento não sabe sobre a forma lógica, passos, tipo de dados, semânticas de indexação, dispositivos ou operações matemáticas.

### Tensor

Combina armazenamento, forma, passos e tipo de dados e garante sua consistência:

- comprimento do armazenamento igual ao número de elementos da forma;
- os passos têm a mesma classificação que a forma;
- os passos descrevem um layout contínuo em linha única na v1;
- o tipo de dados é Float32;
- cada offset acessível está dentro do armazenamento.

## Propriedade e tempo de vida

Tensor v1 possui diretamente o Armazenamento. Não há `Arc`, contagem de referências, estado mutável compartilhado ou visualização emprestada entre a fronteira C ABI.

Visualizações futuras podem usar armazenamento imutável compartilhado como `Arc<Storage>`, mas apenas após a criação, destruição, resultados das operações e o comportamento do ciclo de vida do PHP serem provados com propriedade exclusiva.

## Política de transposição

A existente função de buffer `transpose_f32` materializa dados reordenados e permanece como o comportamento de referência. Tensor v1 deve inicialmente fazer o mesmo e retornar um novo tensor contínuo.

Uma visualização futura pode transpor sem copiar trocando forma e passos:

```text
antes: forma [2, 3], passos [3, 1]
depois:  forma [3, 2], passos [1, 3]
```

Essa otimização é explicitamente fora do escopo de Tensor v1.

## Fronteira opaca da ABI C

O futuro ABI deve expor um tipo incompleto em C, nunca um layout Rust:

```c
typedef struct TransformerTensor TransformerTensor;
```

Funções de ciclo de vida implementadas no T5:

```c
int transformer_tensor_create_f32(
    const float* data,
    const size_t* shape,
    size_t rank,
    TransformerTensor** output
);

void transformer_tensor_destroy(TransformerTensor* tensor);
```

O PHP trata o resultado como um ponteiro opaco. A criação bem-sucedida escreve um único handle exclusivamente proprietário para `output`; a falha deixa-o nulo e retorna um status. Exatamente uma destruição correspondente libera um handle não-nulo, enquanto destruição de nulo é uma operação nula. Destruição dupla e uso após desalocar violam o contrato do chamador. Panics em Rust devem ser capturados antes de cruzarem a fronteira ABI.

Consultas de metadados usam códigos de status e buffers de saída proprietários pelo chamador. A forma é copiada em vez de exposta através de um ponteiro emprestado para o armazenamento Rust.

Operações futuras retornam novos handles opacos:

```c
TransformerTensor* transformer_tensor_matmul(
    const TransformerTensor* a,
    const TransformerTensor* b
);
```

O mecanismo de retorno de erro concreto deve ser projetado antes da implementação; retornar apenas nulo é insuficiente para diagnósticos úteis.

## Fronteira de materialização

Os dados do tensor permanecem nativos em operações. T6 fornece
`transformer_tensor_numel` seguido por `transformer_tensor_copy_data_f32` para depuração, testes, limites de serialização e consumo final do PHP. A cópia de saída nunca escreve quando a capacidade é insuficiente, nunca expõe o armazenamento interno e não consome o handle.

## APIs opacas e paridade

As APIs atuais de buffer permanecem disponíveis durante o desenvolvimento do tensor:

```text
transformer_tensor_add_f32
transformer_matmul_f32
transformer_transpose_f32
transformer_softmax_f32
```

Eles são implementações de referência para paridade:

```text
API de buffer validada ≈ API de handle de tensor
```

Eles não devem ser otimizados embora ou excluídos até que as operações do tensor tenham cobertura de paridade confiável e o projeto explicitamente retirem-os.

## Portões de implementação

Cada portão requer seus próprios testes e revisão antes que o próximo comece:

1. **T1 — DType, Shape, Strides:** completo em Rust puro; sem FFI.
2. **T2 — Storage<Float32>:** completo com propriedade exclusiva de `Vec<f32>`.
3. **T3 — Tensor Rust:** completo como um tensor contínuo Float32 CPU; sem handles.
4. **T4 — Handle opaco:** completo com representação privada em Rust e contrato de ciclo de vida.
5. **T5 — ABI de metadados:** completo com criação, destruição, forma, classificação e tipo de dados.
6. **T6 — Cópia explícita de saída:** completo com consulta de numel e verificações de capacidade.
7. **T7 — Adição de tensor:** completo com paridade da função `add_f32` do buffer.
8. **T8 — Multiplicação matricial de tensor:** completo com paridade da multiplicação matricial básica do buffer.
9. **T9 — Transposição de tensor:** completo como materializada e contínua.
10. **T10 — Softmax de tensor:** completo com comportamento estável em 1D e paridade.

Após T10, o projeto pausa a expansão dos kernels do tensor para uma revisão da arquitetura NN cobrindo Parameter, Module, Linear, Embedding, LayerNorm e GELU. Visualizações, armazenamento compartilhado, tensores não contínuos, broadcasting, múltiplos tipos de dados e kernels otimizados permanecem adiados até que um consumidor concreto os exija.

## Decisões validadas no T5

Antes da implementação começar, confirme:

- valores de status OK `0`, argumento inválido `1`, pânico `2` e buffer insuficiente `3`;
- consultas de forma copiam para o armazenamento do chamador e retornam buffer insuficiente quando a capacidade está abaixo da classificação;
- o discriminante estável do tipo de dados Float32 é `0`;
- falhas na criação deixam o handle de saída nulo;
- criação de escalar e tensores com tamanho zero seguem os contratos de forma em Rust;
- destruição exclusiva do objeto PHP e versão formal da ABI permanecem para trabalho de integração futura.
