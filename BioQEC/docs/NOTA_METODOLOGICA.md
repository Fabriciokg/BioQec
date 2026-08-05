# Nota metodológica: o que o notebook testa

## Unidade simulada

Cada linha do benchmark corresponde a uma janela temporal resumida por:

- início e fim;
- intensidade média de erro;
- modelo nominal dominante;
- indicador de mistura de regimes;
- indicador de presença de ponto de mudança.

Cada janela gera um circuito de memória independente com o mesmo número de ciclos da janela. São amostrados vários *shots* desse circuito.

## Pareamento

O decoder estático e o decoder-oráculo recebem a mesma matriz de eventos de detecção e os mesmos observáveis verdadeiros. A única diferença é o modelo usado para construir os pesos do grafo de matching:

- estático: regime de calibração;
- oráculo: regime nominal da janela.

## Interpretação

Uma diferença de LER entre os dois decoders indica custo de descasamento do modelo. Ela não demonstra ainda que o BioQEC detecta a mudança ou seleciona corretamente um especialista.

## Extensão necessária para o laço online

A versão confirmatória deverá preservar a memória lógica entre ciclos e variar canais de erro dentro do mesmo circuito. Há três caminhos:

1. gerar manualmente o circuito estabilizador e inserir canais por ciclo;
2. transformar recursivamente o circuito Stim, expandindo blocos `REPEAT` e substituindo probabilidades em cada rodada;
3. utilizar registros temporais de hardware e aplicar os decoders ao mesmo fluxo observado.

A opção 1 oferece maior controle e auditoria; a opção 2 reaproveita o circuito gerado, mas exige testes cuidadosos para preservar detectores e observáveis; a opção 3 fornece maior validade externa.
