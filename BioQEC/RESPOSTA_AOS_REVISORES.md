# Resposta técnica aos pontos de revisão

## Formulação matemática

- **Vetor de características:** os oito componentes foram definidos por fórmulas causais, incluindo adjacência espacial, autocorrelação, leitura analógica calibrada e entropia da previsão anterior à prova de leitura. Máscaras substituem imputações quando um canal não está disponível.
- **Função de aptidão:** a soma ponderada foi removida. A promoção agora usa restrições operacionais e fronteira de Pareto, com desempate lexicográfico pré-registrado.
- **Proposições elementares:** foram incorporadas como invariantes ou observações no texto.
- **Novidade:** o CUSUM abre o episódio e a distância à memória apenas o classifica após um período de graça. Um mesmo episódio não pode gerar alarmes duplicados.

## Protocolo experimental

- O artigo principal contém um protocolo compacto de seis páginas. As quatro etapas completas foram movidas para `pre_registro.tex`.
- O MWPM periódico agora possui agenda, janela, contração e equação de atualização explícitas.
- Spitz, Bhardwaj, Sivak, QAdapt e um decoder condicionado por calibração foram elevados a comparadores adaptativos primários; Harmony, Libra e Tesseract atuam como controles fortes.
- OOD é definido pelo gerador a partir de mecanismo, suporte paramétrico, composição e envelope temporal. O limiar do BioQEC é apenas avaliado contra esses rótulos.

## Literatura e apresentação

- A bibliografia passou a 40 itens, incluindo ensembles, modelos recorrentes, estado estruturado, calibração de priors, deriva, dados IBM, execução em tempo real e MoE para QEC.
- O modelo aparece antes da fundamentação.
- O mapeamento bioinspirado foi transformado em figura de visão geral na introdução.
- Os laços rápido e lento foram separados em dois algoritmos.
- As figuras foram precompiladas, renderizadas e verificadas.

## Elementos preservados

Permaneceram centrais: separação causal entre laço rápido e lento, CVaR para eventos severos, recorrência A-B-A, fallback obrigatório, pareamento experimental e critérios explícitos de falsificação.
