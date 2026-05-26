
# K-Armed Bandits

## Introdução

Esta experiência explora o problema clássico Multi-Armed Bandits e o equilíbrio entre exploração e exploração gulosa.

---

## Estrutura Implementada

```text
mia_rl/
  envs/
    k_bandits.py

  agents/
    bandits/

  experiments/
    k_bandits.py

  plots/
    k_bandits.py

  scripts/
    run_kbandits.py
    run_kbandits.bat
```

---

## Descrição do Ambiente

O agente deve selecionar entre múltiplos braços com distribuições de recompensa desconhecidas.

Cada ação produz recompensas estocásticas.

---

## Algoritmos Utilizados

- Epsilon-Greedy
- UCB
- Softmax Exploration
- Optimistic Initialization

---

## Objetivos Experimentais

Comparar:

- capacidade exploratória;
- velocidade de convergência;
- reward acumulada;
- estabilidade.

---

## Resultados

UCB apresentou melhor equilíbrio entre exploração e exploração gulosa.

Estratégias epsilon-greedy apresentaram convergência rápida mas maior sensibilidade ao valor de epsilon.

---

## Conclusões

O problema K-Armed Bandits revelou-se particularmente útil para estudar estratégias fundamentais de exploração em Reinforcement Learning.
