# Documentação

Bem-vindo à documentação do projeto. Aqui você encontrará informações sobre a arquitetura, as principais classes e interfaces, bem como instruções para contribuir com o código-fonte.

## Arquitetura

A arquitetura do projeto é dividida em três partes principais:

1. **API PHP**: Contém contratos de Tensor, rede neural, Transformer, tokenizador, modelo, embedding, serialização e futuras gerações.
2. **BackendInterface**: Isola essas APIs da execução. Módulos de alto nível não sabem sobre FFI, Zend, ponteiros crus ou detalhes internos do Rust.
3. **Runtime Rust**: Possui a execução nativa. O backend FFI e o futuro extensão Zend consumem a mesma ABI estável em C.

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

- **Tensor**: Descreve forma, tipo de dados, dispositivo, armazenamento e futuras operações.
- **Backend**: Define contratos de execução e seleciona uma implementação.
- **NN/Transformer/Model**: Compostos comportamentos tipados de alto nível sem depender dos detalhes de implementação nativa.
- **NativeLibrary**: Carrega a biblioteca FFI, cria Tensors nativos e preserva
  chamadas legadas de buffer.
- **Módulo Rust FFI**: Valida argumentos de ponteiro de nível inferior, previne panics que cruzam a ABI e converte ponteiros crus em fatias seguras.
- **Kernels Rust**: Contêm cálculos seguros sobre fatias e não ponteiros crus.

A API de handles Float32 fornece adição, matmul com dispatcher CPU, transposição
materializada e softmax no último eixo. A bridge PHP `Tensor`/`NativeStorage`
mantém os dados residentes; chamadas de buffer legadas continuam disponíveis
para compatibilidade e paridade.

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
