
# Blackjack

## Introdução

Esta experiência explora a aplicação de algoritmos de Reinforcement Learning ao problema clássico de Blackjack.

O objetivo do agente consiste em aprender uma política capaz de maximizar a recompensa esperada através da interação contínua com o ambiente.

---

## Estrutura Implementada

```text
mia_rl/
  envs/
    blackjack.py

  agents/
    monte_carlo.py
    sarsa.py
    q_learning.py

  experiments/
    blackjack.py

  plots/
    blackjack.py

  scripts/
    run_blackjack_prediction.py
    run_blackjack_prediction.bat
```

---

## Descrição do Ambiente

O ambiente modela o jogo Blackjack segundo uma formulação Markoviana.

O estado é definido por:

- soma atual das cartas do jogador;
- carta visível do dealer;
- existência de usable ace.

### Ações

- Hit
- Stick

### Recompensas

- Vitória: +1
- Derrota: -1
- Empate: 0

---

## Formulação MDP

```
(S, A, P, R)
```

onde:

- S representa os estados possíveis do jogo;
- A representa as ações disponíveis;
- P representa as probabilidades de transição;
- R representa a função de recompensa.

---

## Algoritmos Utilizados

- Monte Carlo Prediction
- Monte Carlo Control
- SARSA
- Q-Learning

---

## Pipeline Experimental

```
Environment
    ↓
Episode Generation
    ↓
Policy Update
    ↓
Evaluation
    ↓
Metrics & Plots
```

---

## Hiperparâmetros

| Parâmetro | Valor |
|---|---|
| Alpha | 0.1 |
| Gamma | 0.99 |
| Epsilon | 0.1 |
| Episodes | 100000 |

---

## Resultados

Os métodos Temporal Difference evidenciaram maior velocidade de convergência relativamente aos métodos Monte Carlo.

Q-Learning demonstrou uma estratégia de exploração mais agressiva e maior capacidade exploratória, enquanto SARSA apresentou comportamento mais estável e políticas mais conservadoras durante o processo de aprendizagem.

---

## Conclusões

O ambiente Blackjack revelou-se particularmente adequado para comparação entre métodos Monte Carlo e Temporal Difference, permitindo observar diferenças claras entre estratégias on-policy e off-policy.
