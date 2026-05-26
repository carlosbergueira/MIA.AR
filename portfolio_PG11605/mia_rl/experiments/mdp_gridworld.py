"""
Lógica de execução dos experimentos do Gridworld.

Executa:
  1. Avaliação da política uniforme aleatória (V^π)
  2. Histórico de avaliação de política (snapshots de V por iteração)
  3. Melhoria gulosa a partir de V^π
  4. Iteração de política completa
  5. Iteração de valor (determinista)
  6. Iteração de valor com dinâmica estocástica (slip 0.8/0.1/0.1)
  7. Iteração de valor com célula armadilha (reward -10)
  8. Comparação de gammas via value iteration
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def run_gridworld(output_dir: Path, gamma: float = 0.9) -> None:
    import matplotlib.pyplot as plt

    # from mia_rl.envs.mdp_gridworld import Gridworld, TrapGridworld                A APAGAR
    from mia_rl.envs.mdp_gridworld import MDP_Gridworld, TrapGridworld
    from mia_rl.agents.planning.mdp_gridworld import (
        uniform_random_policy,
        policy_evaluation,
        policy_evaluation_with_history,
        policy_evaluation_Q,
        policy_improvement,
        policy_iteration,
        greedy_policy_from_V,
        value_iteration,
        value_iteration_stochastic,
        zeros_V,
    )
    from mia_rl.plots.mdp_gridworld import (
        plot_grid,
        plot_policy_iteration_history,
        plot_policy_evaluation_history,
    )

    import numpy as np
    np.set_printoptions(precision=3, suppress=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    # env = Gridworld()             A APAGAR
    env = MDP_Gridworld()

    # ------------------------------------------------------------------
    # 1. Avaliação da política uniforme aleatória
    # ------------------------------------------------------------------
    pi0 = uniform_random_policy(env)
    V_pi0, iters = policy_evaluation(env, pi0, gamma=gamma)
    print(f"Policy evaluation (uniform random) converged in {iters} iterations.")

    fig = plot_grid(env, V_pi0, title="Policy Evaluation: V^π (uniform random π)")
    fig.savefig(output_dir / "policy_evaluation_uniform.png")
    plt.close(fig)
    print(f"Saved: {output_dir / 'policy_evaluation_uniform.png'}")

    # ------------------------------------------------------------------
    # 2. Histórico de avaliação de política (snapshots)
    # ------------------------------------------------------------------
    V_hist, it_hist = policy_evaluation_with_history(env, pi0, gamma=gamma)
    plot_policy_evaluation_history(env, V_hist, it_hist)

    # ------------------------------------------------------------------
    # 3. Avaliação de política no espaço Q
    # ------------------------------------------------------------------
    Q_pi, itq = policy_evaluation_Q(env, pi0, gamma=gamma)
    print(f"Q^π converged in {itq} iterations.")

    V_from_Q = zeros_V(env)
    from mia_rl.envs.mdp_gridworld import ACTIONS
    for (r, c) in env.states():
        s = (r, c)
        if env.is_terminal(s):
            V_from_Q[r, c] = 0.0
        else:
            V_from_Q[r, c] = sum(
                pi0[s][a] * Q_pi[r, c, a_index]
                for a_index, a in enumerate(ACTIONS)
            )
    print(f"max |V_pi - V_from_Q| = {np.max(np.abs(V_pi0 - V_from_Q)):.2e}")

    # ------------------------------------------------------------------
    # 4. Melhoria gulosa a partir de V^π
    # ------------------------------------------------------------------
    pi1_actions, _ = policy_improvement(env, V_pi0, gamma=gamma)
    fig = plot_grid(env, V_pi0, policy=pi1_actions,
                    title="Greedy policy w.r.t. V^π (arrows)")
    fig.savefig(output_dir / "policy_improvement_greedy.png")
    plt.close(fig)
    print(f"Saved: {output_dir / 'policy_improvement_greedy.png'}")

    # ------------------------------------------------------------------
    # 5. Iteração de política completa
    # ------------------------------------------------------------------
    V_star_pi, pi_star_actions, history = policy_iteration(env, gamma=gamma)
    print(f"Policy iteration outer loops: {len(history)}")

    fig = plot_grid(env, V_star_pi, policy=pi_star_actions,
                    title="Policy Iteration: V* and π* (greedy actions)")
    fig.savefig(output_dir / "policy_iteration_final.png")
    plt.close(fig)
    print(f"Saved: {output_dir / 'policy_iteration_final.png'}")

    fig = plot_policy_iteration_history(env, history)
    fig.savefig(output_dir / "policy_iteration_history.png")
    plt.close(fig)
    print(f"Saved: {output_dir / 'policy_iteration_history.png'}")

    # ------------------------------------------------------------------
    # 6. Iteração de valor (determinista)
    # ------------------------------------------------------------------
    V_star, iters_vi = value_iteration(env, gamma=gamma)
    print(f"Value iteration converged in {iters_vi} iterations.")

    pi_star = greedy_policy_from_V(env, V_star, gamma=gamma)
    fig = plot_grid(env, V_star, policy=pi_star,
                    title="Value Iteration: V* and greedy policy")
    fig.savefig(output_dir / "value_iteration_final.png")
    plt.close(fig)
    print(f"Saved: {output_dir / 'value_iteration_final.png'}")

    # ------------------------------------------------------------------
    # 7. Comparação de gammas
    # ------------------------------------------------------------------
    for g in [0.5, 0.9, 0.99]:
        Vg, itg = value_iteration(env, gamma=g)
        pig = greedy_policy_from_V(env, Vg, gamma=g)
        print(f"Gamma = {g} — value iteration iters = {itg}")
        fig = plot_grid(env, Vg, policy=pig,
                        title=f"V* and greedy policy (gamma={g})")
        fname = f"value_iteration_gamma_{str(g).replace('.', '')}.png"
        fig.savefig(output_dir / fname)
        plt.close(fig)
        print(f"Saved: {output_dir / fname}")

    # ------------------------------------------------------------------
    # 8. Célula armadilha (reward -10)
    # ------------------------------------------------------------------
    env_trap = TrapGridworld()
    V_trap, it_trap = value_iteration(env_trap, gamma=0.9)
    pi_trap = greedy_policy_from_V(env_trap, V_trap, gamma=0.9)
    print(f"Trap value iteration iters: {it_trap}")
    fig = plot_grid(env_trap, V_trap, policy=pi_trap,
                    title="V* with trap at (0,2) reward -10")
    fig.savefig(output_dir / "value_iteration_trap.png")
    plt.close(fig)
    print(f"Saved: {output_dir / 'value_iteration_trap.png'}")

    # ------------------------------------------------------------------
    # 9. Dinâmica estocástica (slip 0.8/0.1/0.1)
    # ------------------------------------------------------------------
    V_stoch, its = value_iteration_stochastic(env, gamma=0.9)
    pi_stoch = greedy_policy_from_V(env, V_stoch, gamma=0.9)
    print(f"Stochastic value iteration iters: {its}")
    fig = plot_grid(env, V_stoch, policy=pi_stoch,
                    title="V* with stochastic slip (0.8/0.1/0.1)")
    fig.savefig(output_dir / "value_iteration_stochastic.png")
    plt.close(fig)
    print(f"Saved: {output_dir / 'value_iteration_stochastic.png'}")