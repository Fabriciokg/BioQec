# Relatório de validação do artefato BioQEC

## Testes executáveis sem Stim/PyMatching

A suíte cobre:

- validação das trajetórias e das métricas;
- causalidade e máscaras do vetor de oito características;
- ausência de acesso a dados posteriores ao índice `t`;
- máquina de estados CUSUM--novidade e supressão de alarmes duplicados;
- rótulos OOD independentes do método;
- agenda, contração e transformação de pesos do MWPM periódico;
- viabilidade, dominância e fronteira de Pareto.

Comando:

```bash
pytest -q
```

Resultado na construção do pacote:

```text
31 passed, 1 skipped
```

O teste ignorado é a integração Stim--PyMatching quando essas dependências binárias não estão instaladas no ambiente local. No GitHub Actions, `pip install -e ".[dev]"` instala as dependências e habilita esse teste.

## Invariantes que devem permanecer verdadeiros

1. Alterar observações posteriores a `t` não altera as características calculadas em `t`.
2. O limiar de novidade não participa da construção do rótulo OOD.
3. Um episódio ativo de mudança mantém o mesmo identificador até o reset.
4. O baseline periódico somente recalibra em fronteiras pré-definidas.
5. Nenhum candidato com violação de guardrail entra na fronteira operacional.
6. O decoder-oráculo não é tratado como método implantável.

## Limite da validação atual

A trajetória não estacionária é representada por janelas independentes. A interpretação de atraso de adaptação em uma execução contínua depende de uma extensão com canais variáveis por ciclo ou dados temporais de hardware.
