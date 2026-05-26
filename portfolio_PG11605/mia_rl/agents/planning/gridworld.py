"""
Algoritmos de Programação Dinâmica para o Gridworld.

Funções exportadas:
    uniform_random_policy   — política aleatória uniforme π(a|s) = 1/4
    policy_evaluation       — avaliação iterativa de política
    policy_improvement      — melhoria gulosa (greedy)
    policy_iteration        — iteração de política completa
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from mia_rl.envs.gridworld import ACTIONS, Gridworld


# ---------------------------------------------------------------------------
# Utilitário
# ---------------------------------------------------------------------------

def zeros_V(env: Gridworld) -> np.ndarray:
    return np.zeros((env.n_rows, env.n_cols), dtype=float)


# ---------------------------------------------------------------------------
# Política aleatória uniforme
# ---------------------------------------------------------------------------

def uniform_random_policy(
    env: Gridworld,
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """π(a|s) = 1/4 para estados não-terminais."""
    pi: Dict[Tuple[int, int], Dict[str, float]] = {}
    for s in env.states():
        if env.is_terminal(s):
            pi[s] = {a: 0.0 for a in ACTIONS}
        else:
            pi[s] = {a: 1.0 / len(ACTIONS) for a in ACTIONS}
    return pi


# ---------------------------------------------------------------------------
# Avaliação de política
# ---------------------------------------------------------------------------

def bellman_expectation_update(
    env: Gridworld,
    V: np.ndarray,
    policy: Dict[Tuple[int, int], Dict[str, float]],
    s: Tuple[int, int],
    gamma: float,
) -> float:
    """V(s) ← Σ_a π(a|s) [r(s,a) + γ V(s')]"""
    if env.is_terminal(s):
        return 0.0

    v_new = 0.0
    for a, p in policy[s].items():
        ns, r, done = env.step(s, a)
        v_new += p * (r + gamma * V[ns[0], ns[1]])
    return v_new


def policy_evaluation(
    env: Gridworld,
    policy: Dict[Tuple[int, int], Dict[str, float]],
    gamma: float,
    theta: float = 1e-8,
    max_iters: int = 10_000,
) -> Tuple[np.ndarray, int]:
    """Avaliação iterativa — devolve (V, iterações até convergir)."""
    V = zeros_V(env)

    for it in range(max_iters):
        delta = 0.0
        V_old = V.copy()

        for s in env.states():
            v_new = bellman_expectation_update(env, V_old, policy, s, gamma)
            delta = max(delta, abs(v_new - V[s[0], s[1]]))
            V[s[0], s[1]] = v_new

        if delta < theta:
            return V, it + 1

    return V, max_iters


# ---------------------------------------------------------------------------
# Melhoria de política
# ---------------------------------------------------------------------------

def greedy_action_from_V(
    env: Gridworld,
    V: np.ndarray,
    s: Tuple[int, int],
    gamma: float,
) -> str:
    """argmax_a [r(s,a) + γ V(s')]"""
    best_a: Optional[str] = None
    best_q = -np.inf
    for a in ACTIONS:
        ns, r, done = env.step(s, a)
        q = r + gamma * V[ns[0], ns[1]]
        if q > best_q:
            best_q = q
            best_a = a
    return best_a  # type: ignore[return-value]


def policy_improvement(
    env: Gridworld,
    V: np.ndarray,
    old_policy_actions: Optional[Dict[Tuple[int, int], str]] = None,
    gamma: float = 0.9,
) -> Tuple[Dict[Tuple[int, int], str], bool]:
    """Política gulosa em relação a V.

    Devolve (nova_política_determinista, estável).
    'estável' é False se qualquer acção mudou.
    """
    new_policy_actions: Dict[Tuple[int, int], str] = {}
    stable = True

    for s in env.states():
        if env.is_terminal(s):
            new_policy_actions[s] = "·"
            continue

        best_a = greedy_action_from_V(env, V, s, gamma)
        new_policy_actions[s] = best_a

        if old_policy_actions is not None:
            if old_policy_actions.get(s) != best_a:
                stable = False
        else:
            stable = False  # sem política de referência, não podemos declarar estabilidade

    return new_policy_actions, stable


# ---------------------------------------------------------------------------
# Iteração de política
# ---------------------------------------------------------------------------

def policy_iteration(
    env: Gridworld,
    gamma: float = 0.9,
    theta: float = 1e-8,
    max_outer: int = 100,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], str], list]:
    """Iteração de política completa.

    Devolve (V*, π*, histórico).
    Cada entrada do histórico: (outer_iter, pe_iters, V_cópia, pi_actions_cópia).
    """
    pi_stochastic = uniform_random_policy(env)
    pi_actions: Dict[Tuple[int, int], str] = {
        s: ("·" if env.is_terminal(s) else None)  # type: ignore[dict-item]
        for s in env.states()
    }
    history = []

    for outer in range(max_outer):
        # 1) Avaliar política actual
        V, iters = policy_evaluation(env, pi_stochastic, gamma=gamma, theta=theta)

        # 2) Melhorar: política determinista gulosa
        new_actions, stable = policy_improvement(
            env, V, old_policy_actions=pi_actions, gamma=gamma
        )
        history.append((outer, iters, V.copy(), new_actions.copy()))

        pi_actions = new_actions

        # Converter acções deterministas para representação estocástica π(a|s)
        pi_stochastic = {}
        for s in env.states():
            if env.is_terminal(s):
                pi_stochastic[s] = {a: 0.0 for a in ACTIONS}
            else:
                chosen = pi_actions[s]
                pi_stochastic[s] = {a: (1.0 if a == chosen else 0.0) for a in ACTIONS}

        if stable:
            return V, pi_actions, history

    return V, pi_actions, history