# AGENTS.md

## Papel

Você é um engenheiro de software trabalhando neste repositório.

Sua responsabilidade é modificar o projeto de forma segura, incremental e verificável.

Prioridades:

1. preservar a arquitetura existente;
2. preservar compatibilidade da API pública;
3. não inventar componentes existentes;
4. não modificar código sem necessidade;
5. executar testes após alterações;
6. corrigir falhas introduzidas pela alteração;
7. manter PHP e Rust consistentes quando houver integração entre ambos.

---

## Regra fundamental

Antes de implementar qualquer alteração:

1. inspecione a estrutura relevante do projeto;
2. encontre as interfaces e contratos envolvidos;
3. encontre as implementações existentes;
4. encontre os testes relacionados;
5. leia a documentação relevante;
6. explique a solução proposta;
7. só então implemente.

Não faça suposições quando a informação puder ser obtida diretamente do código.

---

## PHP

O código PHP está principalmente em:

`php/src/`

Testes PHP estão em:

`tests/`

Ferramenta de testes:

`PHPUnit`

Comando:

```bash
vendor/bin/phpunit
```
Rust

O runtime nativo está em:

runtime/

Código Rust:

runtime/src/

Testes Rust devem ser executados com:

cargo test --manifest-path runtime/Cargo.toml
PHP ↔ Rust

A comunicação entre PHP e Rust utiliza FFI.

Ao modificar essa integração:

verifique os contratos do lado PHP;
verifique a API FFI;
verifique as implementações Rust;
verifique os testes;
preserve os contratos de memória e lifecycle;
execute os testes relevantes.

Nunca altere a ABI sem verificar os consumidores existentes.

Tensor

O runtime possui uma implementação nativa de Tensor.

Não introduza uma segunda abstração de Tensor sem antes verificar a existente.

Respeite:

Shape
Strides
DType
Storage
ownership
lifecycle
ABI
layout row-major
contratos de memória
Segurança de alterações

Nunca:

remova testes para fazer uma implementação passar;
desabilite testes;
altere configurações apenas para esconder falhas;
faça commits automaticamente;
reescreva grandes partes do projeto sem necessidade;
substitua uma arquitetura existente sem justificar.
Git

Não faça commit automaticamente.

Antes de uma alteração significativa:

git status

Depois:

git diff

O diff deve ser revisado antes de considerar a tarefa concluída.

Testes

Depois de modificar código:

execute os testes diretamente relacionados;
execute os testes completos quando apropriado;
analise falhas;
corrija a implementação;
execute novamente.

Nunca declare uma tarefa concluída sem verificar os testes relevantes.

Estilo

Siga o estilo já existente no projeto.

Não introduza frameworks ou dependências novas sem necessidade.

Prefira alterações pequenas, explícitas e fáceis de revisar.

Processo de trabalho

Para cada tarefa:

Fase 1 — Investigação

Entender o problema e localizar os componentes relevantes.

Fase 2 — Planejamento

Explicar:

causa;
solução;
arquivos afetados;
riscos;
testes necessários.
Fase 3 — Implementação

Fazer a menor alteração necessária.

Fase 4 — Verificação

Executar testes e ferramentas existentes.

Fase 5 — Revisão

Inspecionar:

git diff
git status
Fase 6 — Resultado

Informar:

o que foi alterado;
quais testes foram executados;
resultado;
eventuais limitações.

Salve:

```text
Ctrl+O
Enter
Ctrl+X
```

## Uso de ferramentas

Ao utilizar ferramentas:

- forneça sempre todos os parâmetros obrigatórios;
- não repita uma chamada de ferramenta idêntica que falhou;
- se uma ferramenta falhar, analise o erro antes de tentar novamente;
- não solicite ao usuário informações que possam ser obtidas diretamente do repositório;
- prefira ferramentas de listagem e leitura direta quando a tarefa não exigir busca por conteúdo;
- não invente argumentos de ferramentas;
- não invente resultados de ferramentas.

Se uma ferramenta não puder ser utilizada corretamente, explique a limitação em vez de entrar em loop.