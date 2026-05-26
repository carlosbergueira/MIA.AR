
# Car Rental

## Introdução

Esta experiência implementa o problema clássico de gestão de aluguer de automóveis utilizando métodos de Dynamic Programming.

O objetivo consiste em determinar políticas ótimas de redistribuição de veículos entre diferentes localizações.

---

## Estrutura Implementada

```text
mia_rl/
  envs/
    car_rental.py

  agents/
    planning/
      car_rental.py

  experiments/
    car_rental.py

  plots/
    car_rental.py

  scripts/
    run_car_rental.py
    run_car_rental.bat
```

---

## Descrição do Ambiente

O ambiente modela:

- pedidos de aluguer;
- devoluções de veículos;
- custos de transferência;
- capacidade máxima de armazenamento por localização.

A dinâmica do ambiente segue distribuições probabilísticas de Poisson.

---

## Formulação MDP

```
(S, A, P, R)
```

---

## Algoritmos Utilizados

- Policy Evaluation
- Policy Iteration
- Value Iteration

---

## Resultados

Os métodos de Dynamic Programming permitiram obter políticas estáveis para o problema de redistribuição de veículos, maximizando a reward esperada ao longo do processo de decisão.

Policy Iteration apresentou maior eficiência computacional relativamente a Value Iteration, demonstrando convergência mais rápida para políticas ótimas.

---

## Conclusões

Este problema permitiu estudar planeamento ótimo em ambientes totalmente modelados e compreender o impacto computacional dos métodos clássicos de Dynamic Programming.
