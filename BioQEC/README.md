# BioQEC - protótipo reprodutível com Stim e PyMatching

Este repositório transforma as definições do artigo BioQEC em componentes executáveis e auditáveis. O foco atual é validar o pipeline de códigos de superfície, representar ruído não estacionário em janelas causais e testar mecanismos básicos de adaptação sem usar informação futura nem rótulos reservados ao oráculo.

## O que está implementado

- geração de circuitos de memória rotacionados com Stim;
- amostragem vetorizada de eventos de detecção e observáveis lógicos;
- decodificação de referência por MWPM com PyMatching;
- trajetórias estacionárias, deriva, mudança abrupta, recorrência e OOD;
- comparação pareada entre decoder estático e decoder-oráculo;
- vetor causal de oito características, com máscaras de disponibilidade;
- coordenação CUSUM--novidade por máquina de estados, sem alarmes duplicados;
- baseline MWPM com agenda explícita de recalibração periódica;
- rótulos OOD definidos pelo gerador, independentemente do limiar do BioQEC;
- seleção multiobjetivo por viabilidade e fronteira de Pareto;
- métricas, notebook reprodutível e testes automatizados.

## Dependências principais

- Python 3.10 ou superior;
- Stim 1.16.0;
- PyMatching 2.4.0;
- NumPy, SciPy, pandas e Matplotlib.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate       # Windows PowerShell
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Teste rápido

```bash
pytest -q
python scripts/run_pilot.py --distance 5 --cycles 40 --window-size 5 --shots 2000
```

O CSV será salvo em `results/pilot.csv`.

## Notebook

Abra `notebooks/BioQEC_Stim_PyMatching.ipynb`. O notebook:

1. verifica as versões e importa o projeto;
2. cria e inspeciona um circuito de memória;
3. estima a taxa de erro lógico com MWPM;
4. constrói trajetórias não estacionárias;
5. compara MWPM estático e oráculo nas mesmas síndromes;
6. aplica um CUSUM exploratório;
7. calcula as oito características causais;
8. demonstra a coordenação entre mudança e novidade;
9. define OOD por suporte de treinamento, sem usar o detector avaliado;
10. demonstra recalibração periódica e seleção por Pareto;
11. exporta resultados e executa os testes.

## Estrutura

```text
circuits/surface_code.py          circuitos, amostragem e metadados
noise/nonstationary.py            trajetórias e janelas
noise/ood.py                      rótulos OOD independentes do método
features/causal_features.py       vetor causal de oito componentes
monitoring/change_detection.py    densidade de síndrome e CUSUM
monitoring/novelty.py             máquina de estados mudança--novidade
decoders/mwpm.py                  decoder MWPM
decoders/adaptive_mwpm.py         agenda e pesos do baseline periódico
selection/pareto.py               restrições, dominância e fronteira de Pareto
experiments/windowed_protocol.py  comparação estático versus oráculo
metrics/evaluation.py             LER, Wilson, Brier e erro por ciclo
scripts/run_pilot.py              execução por linha de comando
notebooks/                         demonstração executável
tests/                             testes unitários e integração
```

## Definições científicas implementadas

### Vetor causal

`extract_causal_features` recebe uma matriz com eixos `(tempo, verificadores)` e um índice `t`. A função acessa somente a janela encerrada em `t` e calcula:

- taxas de detecção em verificadores X e Z;
- frequência de pares consecutivos no mesmo verificador, usada como proxy de falha de medição;
- evidência de vazamento, quando disponível;
- correlação espacial em uma lista de adjacências;
- autocorrelação temporal de defasagem um;
- ambiguidade de leitura analógica calibrada;
- entropia normalizada da distribuição preditiva anterior à prova de leitura.

Componentes sem dados suficientes são acompanhados por uma máscara e não recebem imputação derivada do teste.

### Mudança e novidade

O CUSUM abre um único episódio de mudança. Durante um período de graça, a distância à memória não cria um segundo alarme; ela apenas classifica o episódio como compatível com regime conhecido ou novo para a memória. O reset exige persistência abaixo de um limiar inferior.

### Seleção de candidatos

A implementação principal não soma erro lógico, latência e custo em unidades incompatíveis. Primeiro aplica limites operacionais a latência p99, custo, escalonamento, instabilidade e guardrails; depois calcula a fronteira de Pareto sobre risco lógico, CVaR, Brier Score e tempo de adaptação.

### OOD independente

`noise/ood.py` define OOD a partir do suporte declarado de treinamento: mecanismo não visto, parâmetro fora da faixa, composição retida ou envelope temporal retido. O limiar de novidade do BioQEC é avaliado contra esses rótulos e nunca os define.

## Limitação científica importante

`stim.Circuit.generated()` recebe parâmetros de ruído estáticos para todo o circuito. O protótipo divide a trajetória em janelas independentes; em cada janela, o circuito usa a intensidade média local e as mesmas síndromes são fornecidas aos decoders comparados. Esse desenho valida geração, pareamento, métricas e descasamento dos pesos, mas não equivale a uma única memória lógica contínua com `p(t)` variando ciclo a ciclo.

A avaliação online final exige um circuito customizado que preserve `DETECTOR` e `OBSERVABLE_INCLUDE` enquanto insere canais dependentes do ciclo, ou um fluxo temporal de hardware.

## Reprodutibilidade

- configurações e trajetórias são dataclasses validadas;
- sementes são registradas e derivadas por janela;
- decoders comparados recebem exatamente as mesmas síndromes;
- parâmetros de normalização, limiares e agendas são congelados antes do teste;
- o decoder-oráculo é apenas uma referência superior;
- GitHub Actions executa a suíte em Python 3.11 e 3.13.
