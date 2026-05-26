
# Windy GridWorld

## Introdução

Esta experiência estuda o comportamento de algoritmos de Reinforcement Learning num ambiente GridWorld sujeito a perturbações dinâmicas provocadas pelo vento.

O principal objetivo consiste em analisar estratégias de navegação e convergência em ambientes estocásticos.

---

## Estrutura Implementada

```text
mia_rl/
  envs/
    windy_gridworld.py

  agents/
    sarsa.py
    q_learning.py

  experiments/
    windy_gridworld.py

  plots/
    windy_gridworld.py

  scripts/
    run_windy_gridworld_sarsa.py
    run_windy_gridworld_sarsa.bat
```

---

## Descrição do Ambiente

O ambiente inclui:

- grelha bidimensional;
- estado inicial;
- estado objetivo;
- colunas afetadas pelo vento.

### Ações

- Up
- Down
- Left
- Right

### Recompensas

- Cada movimento: -1
- Objetivo atingido: terminal state

---

## Formulação MDP

```
(S, A, P, R)
```

O vento introduz comportamento não determinístico nas transições entre estados.

---

## Algoritmos Utilizados

- SARSA
- N-Step SARSA
- Q-Learning

---

## Pipeline Experimental

```
Agent
    ↓
Interaction with Environment
    ↓
Policy Update
    ↓
Episode Evaluation
    ↓
Learning Curves
```

---

## Hiperparâmetros

| Parâmetro | Valor |
|---|---|
| Alpha | 0.5 |
| Gamma | 1.0 |
| Epsilon | 0.1 |
| Episodes | 500 |

---

## Resultados

SARSA apresentou trajetórias mais seguras durante o processo de aprendizagem.

Q-Learning revelou convergência mais rápida, embora com comportamento inicialmente mais instável.

---

## Conclusões

O ambiente Windy GridWorld demonstrou claramente o impacto da exploração em ambientes estocásticos e as diferenças comportamentais entre métodos on-policy e off-policy.
