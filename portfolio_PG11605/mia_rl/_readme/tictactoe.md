
# TicTacToe

## Introdução

Esta experiência aborda aprendizagem adversarial utilizando o jogo TicTacToe.

O projeto explora métodos de aprendizagem tabular, policy optimization e planning.

---

## Estrutura Implementada

```text
mia_rl/
  envs/
    tictactoe.py

  agents/
    q_learning.py
    reinforce.py
    mcts.py

  experiments/
    tictactoe.py

  plots/
    tictactoe.py

  scripts/
    run_tictactoe.py
    run_tictactoe_mcts.py
    run_tictactoe_mcts.bat
```

---

## Descrição do Ambiente

O ambiente representa um jogo 3x3 entre dois jogadores.

### Estados

Configuração atual do tabuleiro.

### Ações

Colocação de símbolo numa posição livre.

### Recompensas

- Vitória: +1
- Derrota: -1
- Empate: 0

---

## Algoritmos Utilizados

- Q-Learning
- REINFORCE
- Monte Carlo Tree Search (MCTS)

---

## Pipeline Experimental

```
Self-Play
    ↓
State Evaluation
    ↓
Policy Update
    ↓
Match Evaluation
    ↓
Performance Metrics
```

---

## Resultados

MCTS apresentou melhor capacidade estratégica relativamente aos métodos puramente tabulares.

O uso de self-play permitiu melhoria gradual das políticas aprendidas.

---

## Conclusões

O ambiente TicTacToe permitiu explorar adequadamente conceitos de adversarial learning, planning e policy optimization num contexto relativamente simples mas representativo.