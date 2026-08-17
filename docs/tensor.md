# Tensor no PHP

`Tensor` é o valor central da API PHP. Ele combina um `Shape` com um
`StorageInterface`. No backend FFI, `NativeStorage` possui exclusivamente um
handle opaco para um Tensor `Float32` contíguo, row-major e residente no runtime
Rust.

## Criação e metadata

```php
$tensor = $backend->tensorFromFloat32(
    [1, 2, 3, 4, 5, 6],
    new Shape([2, 3]),
);

$tensor->shape()->dimensions; // [2, 3]
$tensor->ndim();              // 2
$tensor->size();              // 6
```

`tensorFromFloat32()` valida que o array é uma lista numérica e que sua
quantidade de elementos corresponde ao shape. Essa é uma cópia PHP → Rust.

## Operações residentes disponíveis

- `add($other)`: shapes exatamente iguais, sem broadcasting;
- `matmul($other)`: duas matrizes rank-2 compatíveis;
- `transpose()`: transposição materializada rank-2;
- `softmax()`: último eixo de Tensor rank ≥ 1.
- GELU tanh via `Gelu::forward()`: elementwise, shape-preserving e sem parâmetros.

Todas são imutáveis: preservam os inputs e retornam um novo Tensor proprietário.
É possível encadear operações sem criar arrays intermediários:

```php
$output = $input
    ->matmul($weights)
    ->add($residual)
    ->softmax()
    ->transpose();

$result = $output->toFloat32();
```

`toFloat32()` é o único copy-out explícito nesse fluxo. Ele retorna uma lista
achatada em ordem row-major.

Um array PHP não é um buffer `float32` contíguo: cada elemento precisa ser
convertido em um valor Zend e inserido no array. O runtime faz uma única cópia
contígua para CData, mas a construção do array pode custar mais que essa cópia.
Por isso, pipelines devem manter Tensors residentes e chamar `toFloat32()`
somente no resultado realmente consumido pelo PHP.

## Buffer Float32 experimental

`exportFloat32Buffer()` cria uma cópia contígua independente que evita criar um
zval para cada float:

```php
$buffer = $tensor->exportFloat32Buffer();

$buffer->numel();       // número de floats
$buffer->sizeBytes();   // numel * 4
$buffer->shape();
$buffer->dtype();       // DType::Float32
$buffer->valueAt(10);   // materializa somente um valor
$bytes = $buffer->toBytes(); // string Float32 little-endian independente

$buffer->destroy();
```

O CData permanece privado. O buffer recebe uma cópia do Tensor e, por isso,
continua válido depois de `Tensor::destroy()`. Destruir o buffer também não
afeta o Tensor. `toBytes()` não é zero-copy: ele faz uma segunda cópia para uma
string PHP, mas não cria um array nem um zval por elemento.

Essa API é experimental e indicada para consumidores binários, hashing,
serialização ou futuras APIs nativas que aceitem buffers. Se o consumidor
precisar de todos os valores como elementos PHP, `toFloat32()` continua sendo o
contrato correto.

## Softmax e eixo

O softmax opera sempre no último eixo. Para `[128, 768]`, as 128 linhas são
normalizadas independentemente em uma única operação nativa. São aceitos
`softmax()`, `softmax(-1)` e o índice positivo do último eixo. Outros eixos são
rejeitados.

## Lifecycle

Cada `NativeStorage` destrói seu próprio handle quando deixa de existir.
`Tensor::destroy()` permite liberação antecipada e é idempotente. Depois disso,
qualquer tentativa de usar o Tensor lança `LogicException`.

```php
$tensor->destroy();
$tensor->destroy(); // seguro
```

## Limitações atuais

- somente CPU e `Float32`;
- somente storage contíguo row-major;
- sem views, broadcasting ou operações in-place;
- construção pública ocorre por `FfiBackend::tensorFromFloat32()`;
- vários métodos reservados em `Tensor` ainda não possuem kernel.
