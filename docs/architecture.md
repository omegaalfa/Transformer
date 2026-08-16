# Arquitetura

O projeto tem três fronteiras:

1. A API PHP contém contratos de Tensor, rede neural, Transformer, tokenizador, modelo,
   embedding, serialização e futuras gerações.
2. `BackendInterface` isola essas APIs da execução. Módulos de alto nível não sabem sobre FFI, Zend, ponteiros crus ou detalhes internos do Rust.
3. O runtime Rust separado possui a execução nativa. O backend FFI e o futuro extensão Zend consumem a mesma ABI estável em C.

```text
API PHP de alto nível
        |
Abstração de Backend
        |
ABI em C
        |
Runtime Rust construído pelo Cargo
```

## Responsabilidades

- **Tensor** descreve forma, tipo de dados, dispositivo, armazenamento e futuras operações.
- **Backend** define contratos de execução e seleciona uma implementação.
- **NN/Transformer/Model** compostos comportamentos tipados de alto nível sem depender dos detalhes de implementação nativa.
- **NativeLibrary** possui carregamento PHP FFI e chamadas de buffer em C.
- **Módulo Rust FFI** valida argumentos de ponteiro de nível inferior, previne panics que cruzam a ABI e converte ponteiros crus em fatias seguras.
- **Kernels Rust** contêm cálculos seguros sobre fatias e não ponteiros crus.

A API Handle Float32 Tensor agora fornece adição, multiplicação matricial naive, transposição materializada e softmax estável 1D. A ponte alta-level PHP Tensor/NativeStorage ainda pendente; chamadas de buffer legado permanecem disponíveis como referências numéricas.

As propostas de limites, invariantes, modelo de propriedade, direção do handle opaco e portões de implementação em etapas futuras são documentados em
[`native-tensor-design.md`](native-tensor-design.md). Esse documento é um registro de design da fase T1–T10 concluída. As decisões sobre a fronteira NN e módulos são documentadas em [`nn-design.md`](nn-design.md).

## Regra obrigatória de kernel nativo

> Os kernels nunca recebem ponteiros crus. Ponteiros crus existem apenas na fronteira FFI.

Cada operação futura segue a mesma direção:

```text
ponteiros crus em C
    -> runtime/src/ffi
    -> comprimentos validados e fatias seguras
    -> runtime/src/kernels
```

Isso se aplica a implementações futuras de matmul, transposição,
softmax, normalização e atenção. A otimização não debilita essa fronteira.
```