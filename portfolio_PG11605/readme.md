## UM / MIA / RL | Maio 2026
### Portfólio "Reinforcement Learning"
### Carlos Bergueira, PG11605

---

# Introdução

Este repositório reúne um conjunto de implementações, experiências e estudos desenvolvidos no âmbito da unidade curricular de Aprendizagem por Reforço.

O principal objetivo do projeto consistiu na exploração prática de diferentes abordagens de Reinforcement Learning, através da construção de ambientes experimentais, implementação de algoritmos clássicos e análise comparativa do comportamento de aprendizagem dos agentes.

O trabalho inclui métodos baseados em:

- Dynamic Programming;
- Monte Carlo Methods;
- Temporal Difference Learning;
- Function Approximation;
- Policy Gradient;
- Planning;
- Monte Carlo Tree Search;
- Coverage Path Planning.

Além da componente algorítmica, o projeto foi desenvolvido com uma arquitetura modular orientada à reutilização de componentes e separação clara de responsabilidades entre:

- ambientes;
- agentes;
- políticas;
- experiências;
- visualização;
- outputs experimentais.

---

# Objetivos do Projeto

Os principais objetivos deste trabalho foram:

- compreender os fundamentos de Reinforcement Learning;
- implementar algoritmos clássicos de aprendizagem;
- comparar métodos on-policy e off-policy;
- analisar processos de exploração e convergência;
- estudar aproximação funcional em espaços de estados mais complexos;
- desenvolver ambientes experimentais reutilizáveis;
- construir pipelines de treino, avaliação e visualização.

---

# Arquitetura Geral

O projeto encontra-se organizado segundo uma estrutura modular composta por:

```text
mia_rl/

    agents/
    core/
    envs/
    experiments/
    features/
    mdps/
    notebooks/
    outputs/
    plots/
    policies/
    scripts/
```

A arquitetura foi concebida para permitir desacoplamento entre:

- ambientes;
- agentes;
- políticas;
- mecanismos de aprendizagem;
- visualização de resultados;
- pipelines experimentais.

---

# Pipeline Experimental

```text
Environment
    ↓
Agent Interaction
    ↓
Experience Collection
    ↓
Policy Update
    ↓
Training Loop
    ↓
Evaluation
    ↓
Plots / Metrics / Outputs
```

---

# Execução

As diferentes experiências podem ser executadas através dos scripts disponíveis na diretoria `scripts/`.

## Exemplo de execução

```bash
.\scripts\1_run_blackjack_prediction.bat
```

```bash
.\scripts\3_run_car_rental.bat
```

---

# Ambiente de Desenvolvimento

## Criação do ambiente

```bash
conda env create -f environment.yml
```

## Ativação

```bash
conda activate mia_rl
```

---

# Componentes do Projeto

## core/

Define as abstrações fundamentais utilizadas ao longo do projeto.

Inclui:

- Agent
- Environment
- Policy
- Episode
- Transition

Este módulo permite separar:

- lógica de aprendizagem;
- dinâmica dos ambientes;
- estratégias de exploração;
- representação das interações.

---

## envs/

Implementação dos diferentes ambientes de Reinforcement Learning.

Inclui:

- GridWorld;
- Windy GridWorld;
- Blackjack;
- TicTacToe;
- K-Armed Bandits.

Cada ambiente define formalmente um processo de decisão de Markov:

```text
(S, A, P, R)
```

onde:

- `S` representa o espaço de estados;
- `A` representa o conjunto de ações;
- `P` define as probabilidades de transição;
- `R` define a função de recompensa.

---

## agents/

Contém as implementações dos algoritmos de aprendizagem.

### Algoritmos implementados

- Monte Carlo
- SARSA
- N-Step SARSA
- Q-Learning
- TD Learning
- REINFORCE
- Monte Carlo Tree Search (MCTS)
- Linear Function Approximation
- Torch-based Approximation

---

## experiments/

Responsável pelos pipelines experimentais:

- treino;
- avaliação;
- comparação entre algoritmos;
- recolha de métricas;
- geração de resultados.

---

## plots/

Módulo responsável pela visualização dos resultados experimentais.

Inclui:

- learning curves;
- reward evolution;
- heatmaps;
- trajectory visualization;
- convergence analysis;
- policy visualization.

---

# Organização Experimental

As experiências foram desenvolvidas com o objetivo de analisar:

- métodos tabulares vs approximation;
- aprendizagem on-policy vs off-policy;
- Monte Carlo vs Temporal Difference;
- planning vs learning;
- exploração vs exploração gulosa;
- estabilidade e convergência.

Cada caso de estudo inclui:

- análise experimental;
- métricas de desempenho;
- avaliação das políticas aprendidas;
- comparação entre algoritmos;
- visualização dos resultados.

---

# Resultados Obtidos

Os resultados experimentais permitiram observar diferentes comportamentos de aprendizagem consoante os algoritmos utilizados e os ambientes considerados.

De forma geral:

- métodos TD apresentaram convergência mais rápida em ambientes tabulares;
- métodos Monte Carlo demonstraram maior estabilidade em determinados cenários estocásticos;
- técnicas de approximation permitiram generalização para espaços de estados mais complexos;
- estratégias off-policy revelaram maior capacidade exploratória;
- MCTS demonstrou bom desempenho em ambientes adversariais como TicTacToe.

---

# Casos de Estudo

O projeto inclui documentação adicional para diferentes ambientes e experiências:

- `blackjack.md`
- `car_rental.md`
- `tictactoe.md`
- `windy_gridworld.md`
- `kbandits.md`

Cada documento descreve:

- arquitetura do ambiente;
- representação do estado;
- reward shaping;
- algoritmos utilizados;
- métricas experimentais;
- resultados obtidos;
- análise comparativa.

---

# Tecnologias Utilizadas

- Python
- NumPy
- Matplotlib
- PyTorch

---

# Considerações Finais

O projeto permitiu consolidar conceitos fundamentais de Reinforcement Learning através da implementação prática de múltiplos algoritmos e ambientes experimentais.

A arquitetura modular adotada procurou aproximar o trabalho de uma pequena framework experimental, promovendo reutilização, extensibilidade e facilidade de comparação entre diferentes abordagens de aprendizagem.
