# Revisão da Arquitetura de Redes Neurais (NN-R1)

Status: **revisão de design aprovada; nenhum desenvolvimento de implementação de rede neural é autorizado por este documento**.

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

A assinatura atual da `Module::forward(Tensor): Tensor` é muito restritiva como uma interface universal. Embedding começa com IDs de token, e mais tarde a atenção também aceita máscaras. NN-1 deve tornar `Module` o contrato de introspecção/composição:

```text
parâmetros() -> map<string, Parameter>
módulos()    -> map<string, Module>
```

Módulos concretos declaram seus próprios métodos `forward` tipados. Componentes Tensor-to-Tensor como `Linear`, `LayerNorm` e `GELU` mantêm
`forward(Tensor): Tensor`. Mapas explícitos são preferidos sobre reflexão ou descoberta de propriedades mágicas. A navegação recursiva constrói um dicionário de estado determinístico.

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

A adição Tensor atual estrita não pode adicionar um bias `[out_features]` a
`[rows, out_features]`, e a multiplicação matmul aceita apenas rango 2. As entradas do Transformer eventualmente terão dimensões de leitura como
`[batch, sequence, hidden]`. NN-2 deve, portanto, projetar uma única operação nativa Linear específica que:

- exija que a última dimensão da entrada seja igual a `input_features`;
- trate as dimensões anteriores como linhas sem expor uma cópia PHP;
- multiplica por peso de forma `[input_features, output_features]`;
- adiciona opcionalmente bias de forma `[output_features]` dentro do mesmo caminho nativo;
- preserva todas as dimensões anteriores e substitui apenas a última;
- retorna um novo Tensor contínuo.

Isso evita adicionar prematuramente semântica de broadcasting ou reshape/view genéricos.
Os testes de referência ainda comparam sua parte de multiplicação com o núcleo básico.

### Embedding

```text
Embedding
  peso Parameter [vocabulary_size, dimensions]
  IDs de token
  coleta linhas
```

IDs de token não devem ser codificados como Float32. Porque a saída do tokenizador origina-se em PHP e Tensor v1 intencionalmente suporta apenas Float32, NN-3 deve inicialmente definir uma fronteira dedicada de pesquisa de embedding aceitando IDs de token inteiro validados e um handle de peso Float32. Ele retorna um Tensor nativo Float32 e realiza apenas uma transferência PHP-to-native no limite de entrada do modelo. Um tipo de dado Tensor genérico inteiro é adiado até que outro consumidor real justifique isso.

A validação inclui IDs não negativos, limites de vocabulário, forma de saída e conversão de largura de inteiro. Um operador de coleta geral-purpose não é necessário antes deste contrato seja provado.

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

NN-5 apresenta uma única operação explícita Float32 GELU de referência e operações de handle Tensor. A revisão deve escolher e documentar a exata ou a aproximação tanh antes da implementação; a configuração/modelo/pesos devem usar a variante correspondente. GELU é elemento-wise, imutável, preservando a forma e requer paridade contra um referencial confiável.

## Portões

```text
NN-R1  revisão de arquitetura da rede neural                              COMPLETO
NN-0   ponte nativa do Tensor no PHP                                  COMPLETO
NN-1   contratos de introspecção e composição para parâmetro e módulo       PENDENTE
NN-2   Linear com projeção na última dimensão nativa                      PENDENTE
NN-3   Embedding com pesquisa de token inteiro validada                    PENDENTE
NN-4   LayerNorm estável                                                   PENDENTE
NN-5   GELU                                                              PENDENTE
NN-R2  revisão antes da atenção                                          PENDENTE
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
- implementação da atenção antes de NN-R2.
