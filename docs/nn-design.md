# Revisão da Arquitetura de Redes Neurais (NN-R1)

Status: **NN-1 a NN-5 implementados; componentes posteriores permanecem pendentes**.

A fase Tensor T1–T10 prova a propriedade nativa, ciclo de vida, metadados,
materialização e execução ponte-a-ponte para adição, multiplicação matricial, transposição e
softmax estável. NN-R1 define como essas capacidades se tornam componentes de rede neural sem transformar o tempo de execução em uma coleção não planejada de operadores.

## Ponte do handle PHP

O ABI C e a bridge PHP suportam handles nativos de Tensor:

- `NativeLibrary` cria Tensors e também preserva as APIs legadas de buffer;
- `NativeStorage` possui e destrói exclusivamente cada handle;
- metadata e operações PHP delegam diretamente à Tensor API;
- `toFloat32()` torna a materialização uma fronteira explícita.

Assim, o gate **NN-0 ponte nativa de Tensor** está concluído. `Linear` e os
demais módulos podem compor operações residentes sem converter resultados
intermediários para arrays PHP.

## Posicionamento das responsabilidades

A arquitetura é deliberadamente dividida:

```text
PHP
  composição de modelos
  metadados de módulo e parâmetro
  nomes de parâmetros e dicionários de estado
  validação e exceções de interface do usuário
  saída do tokenizador e configuração do modelo

Rust
  armazenamento e ciclo de vida do Tensor
  invariantes de forma/dtype
  execução numérica
  validação específica da operação no ABI C
  novos tensores resultantes alocados
```

`Linear`, `Embedding`, `LayerNorm`, `GELU` e blocos Transformer posteriores são objetos de composição PHP. Seu trabalho numérico permanece em Rust. Eles nunca devem copiar dados nativos do Tensor para o PHP entre operações.

## Contrato de parâmetro

`Parameter` não é uma segunda implementação de Tensor. Ele envolve um único `Tensor` PHP cujo armazenamento pode possuir um handle nativo, mais metadados do modelo:

```text
Parameter
  nome: nome local do módulo
  tensor: Tensor
  treinável: bool
```

Regras para v1:

- forma, passos, dtype, dispositivo e armazenamento existem apenas em Tensor;
- Parameter nunca duplica ou muta o armazenamento de Tensor;
- `treinável` é metadados descritivos; v1 é apenas inferência;
- não há gradiente, otimizador, gráfico autograd ou estado de treinamento;
- os parâmetros podem ser lidos por vários módulos porque as operações do Tensor são imutáveis, enquanto a destruição exclusiva do handle nativo permanece com `NativeStorage`;
- a navegação dos módulos produz nomes qualificados como `encoder.layer0.weight`;
  Parameter armazena apenas seu nome local.

## Contrato de módulo

`Module` é o contrato explícito de introspecção/composição:

```text
parâmetros() -> map<string, Parameter>
módulos()    -> map<string, Module>
```

`TensorModule extends Module` adiciona `forward(Tensor): Tensor` para módulos
Tensor-to-Tensor como `Linear`, `LayerNorm` e `GELU`. Módulos com outra
fronteira declaram uma operação tipada própria; `Embedding` permanece `Module`
e recebe IDs inteiros em `forwardTokenIds()`. Mapas explícitos são preferidos
sobre reflexão ou descoberta de propriedades mágicas.

Módulos folha recebem explicitamente um `Runtime` e despacham através do seu backend.
As entradas, parâmetros e resultados devem usar backends compatíveis, dispositivos e dtypes.
Os objetos PHP não sabem nem ponteiros FFI nem layouts Rust.

## Propriedade de handle nativo no PHP

NN-0 deve introduzir um proprietário PHP para cada handle nativo, conceitualmente
`NativeStorage`:

- a construção recebe um handle opaco vivo;
- copiar uma referência do objeto PHP não duplica o handle;
- seu destruidor chama a destruição nativa no máximo uma vez;
- um handle liberado/movido não pode ser usado novamente;
- resultados das operações criam novos proprietários `NativeStorage`;
- metadados são armazenados apenas se forem imutáveis e verificados do ABI;
- `toArray()` é a única fronteira explícita de materialização de dados.

As operações de backend aceitam tensores PHP e delegam operações de handle nativo para
`NativeLibrary`. Nenhuma matriz PHP intermediária é permitida em uma passagem de frente para trás da rede neural.

## Mapa de dependência de módulos

### Linear

```text
Linear
  peso Parameter
  opcional bias Parameter
  projeção final na última dimensão nativa
```

A adição Tensor estrita não pode adicionar um bias `[out_features]` a
`[rows, out_features]`, e a multiplicação matmul aceita apenas rango 2. As entradas do Transformer eventualmente terão dimensões de leitura como
`[batch, sequence, hidden]`. NN-2 implementa uma única operação nativa Linear específica que:

- exija que a última dimensão da entrada seja igual a `input_features`;
- trate as dimensões anteriores como linhas sem expor uma cópia PHP;
- multiplica por peso de forma `[input_features, output_features]`;
- adiciona opcionalmente bias de forma `[output_features]` dentro do mesmo caminho nativo;
- preserva todas as dimensões anteriores e substitui apenas a última;
- retorna um novo Tensor contínuo.

Isso evita adicionar prematuramente semântica de broadcasting ou reshape/view genéricos.
Os testes de referência ainda comparam sua parte de multiplicação com o núcleo básico.

`Linear` possui `weight: Parameter` obrigatório e `bias: ?Parameter`. Esses
objetos permanecem residentes durante a vida do módulo. `forward()` recebe o
input por chamada, delega `linear_last_dim` ao backend e retorna um Tensor novo;
nenhum input, temporário ou output anterior é armazenado no módulo.

O kernel CPU achata conceitualmente todas as dimensões anteriores em M, sem
alterar ou copiar metadata do input, reutiliza o dispatcher matmul existente e
aplica o bias por linha sem expandi-lo fisicamente. Shapes de rank 1 ou superior
são aceitos; a última dimensão deve coincidir com `input_features`.

### Embedding

```text
Embedding
  peso Parameter [vocabulary_size, dimensions]
  IDs de token
  coleta linhas
```

`Embedding` possui `weight: Parameter` residente, Float32, contíguo e row-major
com shape `[vocabulary_size, dimensions]`. Sua API é
`forwardTokenIds(list<int>, Shape([B,S]))`. IDs são achatados em row-major,
atravessam a ABI como `int64_t` e nunca são convertidos para Float32. O output é
um novo Tensor Float32 `[B,S,D]` com Storage independente.

PHP valida lista, rank, produto `B*S`, quantidade e intervalo. A bridge cria
somente um CData `int64_t` temporário; a FFI Rust repete as validações de
segurança e chama um kernel sobre slices seguros. Erro não consome o weight nem
publica output parcial. `[0,S]`, `[B,0]` e `[0,0]` são válidos e não acessam o
weight. `TokenizationResult` continua descrevendo uma sequência; batching e
padding permanecem fora do tokenizer.

### LayerNorm

```text
LayerNorm
  entrada Tensor
  gamma Parameter
  beta Parameter
  epsilon
  normalização estável sobre a última dimensão
```

Em vez de primeiro expor operadores públicos relacionados à média, variância, subtração, multiplicação e
raiz quadrada inversa, NN-4 deve começar com um kernel de LayerNorm de referência seguro. Ele
calcula a média e a variância estável sobre a última dimensão, aplica epsilon,
gamma e beta, preserva as dimensões anteriores e retorna um novo Tensor.
Primitivos kernels devem ser extraídos mais tarde apenas quando múltiplos consumidores precisarem deles.

### GELU

NN-5 implementa exclusivamente a aproximação tanh canônica:

```text
0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
```

Input e output usam Storage Float32; cada elemento é convertido para Float64,
calculado integralmente nessa precisão, validado e convertido uma vez para
Float32. GELU é elementwise, stateless, não possui Parameter e preserva
exatamente qualquer shape, incluindo rank zero e Tensors vazios. NaN, +Inf,
-Inf e resultados não finitos são rejeitados antes da publicação do output.
Paridade com a referência PHP double usa `atol=1e-6` e `rtol=1e-6`.

## Portões

```text
NN-R1  revisão de arquitetura da rede neural                              COMPLETO
NN-0   ponte nativa do Tensor no PHP                                  COMPLETO
NN-1   contratos de introspecção e composição para parâmetro e módulo       COMPLETO
NN-2   Linear com projeção na última dimensão nativa                      COMPLETO
NN-3   Embedding com pesquisa de token inteiro validada                    COMPLETO
NN-4   LayerNorm estável                                                   COMPLETO
NN-5   GELU tanh                                                          COMPLETO
NN-R2  revisão antes da atenção                                          COMPLETO
NN-6   MultiHeadAttention não causal                                     COMPLETO
NN-R3  revisão antes do TransformerBlock                                 COMPLETO
NN-7   FeedForward e TransformerBlock Pre-Norm                           COMPLETO
```

Cada portão de implementação requer sua própria aprovação, testes e atualização de documentação.
NN-R2 deve revisar a orientação Q/K/V, formas de batch/sequence/head,
máscaras, escalas e eixo do softmax antes de qualquer implementação de atenção.

## Explicitamente adiadas

- treinamento, gradientes, autograd e otimizadores;
- broadcasting genérico;
- coleta geral-purpose antes que a Embedding provar suas necessidades;
- reduções gerais apenas para montar LayerNorm;
- visualizações e tensores não contínuos;
- tipos de dados adicionais apenas para representar a saída do tokenizador;
- SIMD, BLAS, GPU e blocos Transformer fusos;
- causal Attention, KV cache, RoPE e positional encoding;
- FeedForward e TransformerBlock antes de seus próprios gates.

## GPU Readiness

O `Linear` PHP depende somente de `Runtime->backend()->linear()`, portanto não
contém OpenBLAS, ponteiros FFI ou decisões de CPU. Um backend GPU futuro deverá
oferecer Linear last-dim, além das operações usadas pelos módulos posteriores
(normalizações, ativações e atenção), sem materializar resultados no PHP.

Antes de CUDA, Tensor/Storage precisam representar explicitamente device e
storage CPU/GPU, e os backends devem rejeitar operações entre dispositivos sem
uma transferência explícita. Pesos devem ser carregados uma vez no Storage GPU
e permanecer na VRAM durante a vida do módulo. Input, weight, bias, temporários
e output de uma execução devem permanecer no mesmo dispositivo para evitar
cópias CPU↔GPU por operação.

O backend selecionado pelo Runtime decide entre o dispatcher CPU/OpenBLAS e um
futuro kernel CUDA/cuBLAS. OpenBLAS continua específico do backend CPU; streams,
handles cuBLAS, alocadores de VRAM e seleção de kernels continuam específicos do
backend GPU. Shape, dtype, ownership de Parameter e a semântica last-dim são
contratos compartilhados e não devem depender de nenhum desses detalhes.

## NN-4 — LayerNorm de inferência

`LayerNorm` mantém `weight` (gamma) e `bias` (beta) como `Parameter` residentes
de shape `[D]`. `forward()` normaliza somente a última dimensão de um Tensor
Float32 contíguo e retorna storage novo, sem Tensors intermediários:
`gamma * (x - mean) / sqrt(variance + epsilon) + beta`. O runtime usa Welford
com acumuladores Float64 e variância populacional. `epsilon` é `1e-5` por
default e deve continuar positivo e finito após conversão para Float32.

Rank zero, `D=0`, shapes incompatíveis e qualquer NaN/Inf são rejeitados.
Dimensões externas vazias são válidas quando `D>0` e preservam exatamente o
shape. Gamma, beta e input permanecem imutáveis; falhas não publicam output.

## NN-6 — MultiHeadAttention de inferência

`MultiHeadAttention` é um `Module`, não um `TensorModule`, porque sua operação
aceita uma `AttentionMask` opcional além do input. O módulo possui quatro
`Linear(D,D)` sem bias e residentes, expostos em ordem determinística por
`modules()` como `q_proj`, `k_proj`, `v_proj` e `out_proj`. A navegação produz
os nomes qualificados `q_proj.weight`, `k_proj.weight`, `v_proj.weight` e
`out_proj.weight`; o módulo não duplica esses Parameters em `parameters()`.

O input e o output têm shape `[B,S,D]`, somente rank 3. `H>0`, `D>0` e
`D % H == 0`; `Dh=D/H`. O runtime projeta Q/K/V, organiza-os logicamente como
`[B,H,S,Dh]`, calcula scores `[B,H,S,S]`, aplica a escala Float32 obtida de
`1/sqrt(Float64(Dh))`, normaliza no último eixo, combina V, recompõe
`[B,S,D]` e aplica `out_proj`. Todas as transformações permanecem internas à
operação dedicada; Tensor não ganhou reshape, permute, views ou batched matmul.

`AttentionMask` é um value object PHP com `list<bool>` row-major e Shape
`[B,S]`. `true` permite a key e `false` a exclui da normalização. Não existe
broadcasting, máscara por head, representação Float32, causalidade ou `-Inf`.
Um batch sem nenhuma key válida é erro. A bridge converte a máscara apenas
durante a chamada para `uint8_t`, onde 0/1 preservam a mesma semântica.

`[B,0,D]`, `[0,S,D]` e `[0,0,D]` retornam output vazio idêntico sem executar
projeções ou reduções. Input, weights, intermediários e output não finitos são
rejeitados sem publicação parcial. Inputs, máscara e weights são emprestados;
o output possui Storage independente e nenhum estado de execução permanece no
módulo. A referência PHP usa doubles; no caso controlado com e sem máscara, os
máximos observados foram `1.034e-7` absoluto e `9.528e-7` relativo. O gate usa
`atol=1e-5` e `rtol=1e-5`, conservadores para os casos cobertos e sem promessa
de paridade bitwise.

## NN-7 — FeedForward e TransformerBlock Pre-Norm

`FeedForward` é um `Module` composto por três módulos residentes, expostos
nesta ordem por `modules()`: `input_projection`, `activation` e
`output_projection`. A primeira `Linear` executa `D -> I` com weight `[D,I]`
e bias `[I]`; a ativação é obrigatoriamente a `Gelu` tanh de NN-5; a segunda
`Linear` executa `I -> D` com weight `[I,D]` e bias `[D]`. `D` é
`hiddenSize`, `I` é `intermediateSize`, e nenhuma projeção biasless ou ativação
configurável faz parte deste gate.

`TransformerBlock` é um `Module`, não um `TensorModule`, porque seu forward
recebe `?AttentionMask`. Ele possui exatamente `norm1`, `attention`, `norm2` e
`feed_forward`, nessa ordem pública, e executa Pre-Norm:

```text
a = input + attention(norm1(input), mask)
output = a + feed_forward(norm2(a))
```

Os residuals usam `Tensor::add()` estrito com shapes idênticos e sem
broadcasting. A máscara `[B,S]` é temporária e alcança somente Attention.
Inputs e outputs são `[B,S,D]`; `[0,S,D]`, `[B,0,D]` e `[0,0,D]` são válidos
quando `D>0`. Parameters permanecem nos módulos folha; Block e FeedForward
retornam `parameters()=[]` e não armazenam inputs, outputs ou temporários.

A composição ocorre integralmente em PHP sobre Tensors nativos residentes.
Não existem ABI ou kernel fused de FeedForward/TransformerBlock, integração ao
Graph Executor, dropout, causalidade, RoPE, positional encoding ou GPU neste
gate.
