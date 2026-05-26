"""
Algoritmos de Programação Dinâmica para o Gridworld.

Funções exportadas:
    zeros_V                         — inicializa V a zeros
    zeros_Q                         — inicializa Q a zeros
    uniform_random_policy           — política aleatória uniforme π(a|s) = 1/4
    bellman_expectation_update      — backup de Bellman para avaliação de política
    bellman_optimality_update       — backup de Bellman óptimo (max_a)
    policy_evaluation               — avaliação iterativa de política
    policy_evaluation_with_history  — avaliação iterativa com histórico de V
    policy_evaluation_Q             — avaliação de política no espaço Q
    greedy_policy_from_V            — política determinista gulosa em relação a V
    policy_improvement              — melhoria gulosa com detecção de estabilidade
    policy_iteration                — iteração de política completa
    value_iteration                 — iteração de valor (Bellman óptimo)
    value_iteration_stochastic      — iteração de valor com dinâmica estocástica (slip)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# from mia_rl.envs.mdp_gridworld import ACTIONS, Gridworld          A APAGAR
from mia_rl.envs.mdp_gridworld import ACTIONS, MDP_Gridworld


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

# def zeros_V(env: Gridworld) -> np.ndarray:                    A APAGAR
def zeros_V(env: MDP_Gridworld) -> np.ndarray:
    """Inicializa a função de valor V a zeros (shape: n_rows × n_cols)."""
    return np.zeros((env.n_rows, env.n_cols), dtype=float)


# def zeros_Q(env: Gridworld) -> np.ndarray:                    A APAGAR
def zeros_Q(env: MDP_Gridworld) -> np.ndarray:
    """Inicializa a função de valor Q a zeros (shape: n_rows × n_cols × |A|)."""
    return np.zeros((env.n_rows, env.n_cols, len(ACTIONS)), dtype=float)


# ---------------------------------------------------------------------------
# Política aleatória uniforme
# ---------------------------------------------------------------------------

def uniform_random_policy(
    # env: Gridworld,                   # A APAGAR
    env: MDP_Gridworld,
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
# Backups de Bellman
# ---------------------------------------------------------------------------

def bellman_expectation_update(
    # env: Gridworld,           # A APAGAR
    env: MDP_Gridworld,
    V: np.ndarray,
    policy: Dict[Tuple[int, int], Dict[str, float]],
    state: Tuple[int, int],
    gamma: float,
) -> float:
    """V(s) ← Σ_a π(a|s) [r(s,a) + γ V(s')]"""
    if env.is_terminal(state):
        return 0.0
    v_new = 0.0
    for a, p in policy[state].items():
        ns, r, done = env.step(state, a)
        v_new += p * (r + gamma * V[ns[0], ns[1]])
    return v_new


def bellman_optimality_update(
    # env: Gridworld,           # A APAGAR
    env: MDP_Gridworld,
    V: np.ndarray,
    s: Tuple[int, int],
    gamma: float,
) -> float:
    """V(s) ← max_a [r(s,a) + γ V(s')]"""
    if env.is_terminal(s):
        return 0.0
    best = -np.inf
    for a in ACTIONS:
        ns, r, done = env.step(s, a)
        best = max(best, r + gamma * V[ns[0], ns[1]])
    return best


# ---------------------------------------------------------------------------
# Avaliação de política
# ---------------------------------------------------------------------------

def policy_evaluation(
    # env: Gridworld,           # A APAGAR
    env: MDP_Gridworld,
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
        for state in env.states():
            v_new = bellman_expectation_update(env, V_old, policy, state, gamma)
            delta = max(delta, abs(v_new - V[state[0], state[1]]))
            V[state[0], state[1]] = v_new
        if delta < theta:
            return V, it + 1
    return V, max_iters


def policy_evaluation_with_history(
    # env: Gridworld,           # A APAGAR
    env: MDP_Gridworld,
    policy: Dict[Tuple[int, int], Dict[str, float]],
    gamma: float,
    theta: float = 1e-8,
    max_iters: int = 10_000,
) -> Tuple[List[np.ndarray], List[int]]:
    """Avaliação iterativa com histórico de snapshots de V a cada iteração."""
    V = zeros_V(env)
    V_history = [V.copy()]
    iters_history = [0]
    for it in range(max_iters):
        delta = 0.0
        V_old = V.copy()
        for state in env.states():
            v_new = bellman_expectation_update(env, V_old, policy, state, gamma)
            delta = max(delta, abs(v_new - V[state[0], state[1]]))
            V[state[0], state[1]] = v_new
        V_history.append(V.copy())
        iters_history.append(it + 1)
        if delta < theta:
            return V_history, iters_history
    return V_history, iters_history


def policy_evaluation_Q(
    # env: Gridworld,           # A APAGAR
    env: MDP_Gridworld,
    policy: Dict[Tuple[int, int], Dict[str, float]],
    gamma: float,
    theta: float = 1e-8,
    max_iters: int = 10_000,
) -> Tuple[np.ndarray, int]:
    """Avaliação de política no espaço Q(s,a). Devolve (Q, iterações)."""
    Q = zeros_Q(env)
    for it in range(max_iters):
        delta = 0.0
        Q_old = Q.copy()
        for (r, c) in env.states():
            s = (r, c)
            if env.is_terminal(s):
                Q[r, c, :] = 0.0
                continue
            for a_index, a in enumerate(ACTIONS):
                ns, reward, done = env.step(s, a)
                nr, nc = ns
                exp_next = sum(policy[ns][a2] * Q_old[nr, nc, aj]
                               for aj, a2 in enumerate(ACTIONS))
                q_new = reward + gamma * exp_next
                delta = max(delta, abs(q_new - Q[r, c, a_index]))
                Q[r, c, a_index] = q_new
        if delta < theta:
            return Q, it + 1
    return Q, max_iters


# ---------------------------------------------------------------------------
# Melhoria de política
# ---------------------------------------------------------------------------

def greedy_policy_from_V(
    # env: Gridworld,           # A APAGAR
    env: MDP_Gridworld,
    V: np.ndarray,
    gamma: float,
) -> Dict[Tuple[int, int], str]:
    """Política determinista gulosa: π(s) = argmax_a [r(s,a) + γ V(s')]"""
    pi_greedy: Dict[Tuple[int, int], str] = {}
    for s in env.states():
        if env.is_terminal(s):
            pi_greedy[s] = "·"
            continue
        best_a: Optional[str] = None
        best_q = -np.inf
        for a in ACTIONS:
            ns, r, done = env.step(s, a)
            q = r + gamma * V[ns[0], ns[1]]
            if q > best_q:
                best_q = q
                best_a = a
        pi_greedy[s] = best_a  # type: ignore[assignment]
    return pi_greedy


def policy_improvement(
    # env: Gridworld,           # A APAGAR
    env: MDP_Gridworld,
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
        best_a: Optional[str] = None
        best_q = -np.inf
        for a in ACTIONS:
            ns, r, done = env.step(s, a)
            q = r + gamma * V[ns[0], ns[1]]
            if q > best_q:
                best_q = q
                best_a = a
        new_policy_actions[s] = best_a  # type: ignore[assignment]
        if old_policy_actions is not None:
            if old_policy_actions.get(s) != best_a:
                stable = False
        else:
            stable = False
    return new_policy_actions, stable


# ---------------------------------------------------------------------------
# Iteração de política
# ---------------------------------------------------------------------------

def policy_iteration(
    # env: Gridworld,           # A APAGAR
    env: MDP_Gridworld,
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
        V, iters = policy_evaluation(env, pi_stochastic, gamma=gamma, theta=theta)
        new_actions, stable = policy_improvement(
            env, V, old_policy_actions=pi_actions, gamma=gamma
        )
        history.append((outer, iters, V.copy(), new_actions.copy()))
        pi_actions = new_actions

        pi_stochastic = {}
        for s in env.states():
            if env.is_terminal(s):
                pi_stochastic[s] = {a: 0.0 for a in ACTIONS}
            else:
                chosen = pi_actions[s]
                pi_stochastic[s] = {a: (1.0 if a == chosen else 0.0) for a in ACTIONS}

        if stable:
            return V, pi_actions, history

    return V, pi_actions, history  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Iteração de valor
# ---------------------------------------------------------------------------

def value_iteration(
    # env: Gridworld,           # A APAGAR
    env: MDP_Gridworld,
    gamma: float,
    theta: float = 1e-8,
    max_iters: int = 10_000,
) -> Tuple[np.ndarray, int]:
    """Iteração de valor (Bellman óptimo). Devolve (V*, iterações)."""
    V = zeros_V(env)
    for it in range(max_iters):
        delta = 0.0
        V_old = V.copy()
        for s in env.states():
            v_new = bellman_optimality_update(env, V_old, s, gamma)
            delta = max(delta, abs(v_new - V[s[0], s[1]]))
            V[s[0], s[1]] = v_new
        if delta < theta:
            return V, it + 1
    return V, max_iters


# ---------------------------------------------------------------------------
# Iteração de valor estocástica (slip 0.8/0.1/0.1)
# ---------------------------------------------------------------------------

_LEFT_OF  = {"U": "L", "D": "R", "L": "D", "R": "U"}
_RIGHT_OF = {"U": "R", "D": "L", "L": "U", "R": "D"}


def _expected_backup_stochastic(
    # env: Gridworld,           # A APAGAR
    env: MDP_Gridworld,
    V: np.ndarray,
    s: Tuple[int, int],
    a: str,
    gamma: float,
) -> float:
    """Q(s,a) com slip: 0.8 direcção pretendida, 0.1 esquerda, 0.1 direita."""
    if env.is_terminal(s):
        return 0.0
    outcomes = [(a, 0.8), (_LEFT_OF[a], 0.1), (_RIGHT_OF[a], 0.1)]
    exp = 0.0
    for a_eff, p in outcomes:
        ns, r, done = env.step(s, a_eff)
        exp += p * (r + gamma * V[ns[0], ns[1]])
    return exp


def value_iteration_stochastic(
    # env: Gridworld,           # A APAGAR
    env: MDP_Gridworld,
    gamma: float,
    theta: float = 1e-8,
    max_iters: int = 10_000,
) -> Tuple[np.ndarray, int]:
    """Iteração de valor com dinâmica estocástica (slip 0.8/0.1/0.1).

    Devolve (V*, iterações).
    """
    V = zeros_V(env)
    for it in range(max_iters):
        delta = 0.0
        V_old = V.copy()
        for s in env.states():
            if env.is_terminal(s):
                V[s[0], s[1]] = 0.0
                continue
            best = max(
                _expected_backup_stochastic(env, V_old, s, a, gamma)
                for a in ACTIONS
            )
            delta = max(delta, abs(best - V[s[0], s[1]]))
            V[s[0], s[1]] = best
        if delta < theta:
            return V, it + 1
    return V, max_iters