"""
Funções de visualização para o Gridworld.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# from mia_rl.envs.mdp_gridworld import ARROW, Gridworld                A APAGAR
from mia_rl.envs.mdp_gridworld import ARROW, MDP_Gridworld


def plot_grid(
    # env: Gridworld,           # A APAGAR
    env: MDP_Gridworld,
    V: np.ndarray,
    policy: Optional[Dict[Tuple[int, int], str]] = None,
    title: str = "",
    value_fmt: str = "{:.2f}",
    ax=None,
) -> plt.Figure:
    """Desenha a grelha com valores V e (opcionalmente) setas da política.

    Se ax for None, cria uma nova figura e devolve-a.
    Estados terminais têm fundo sombreado.
    """
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    ax.set_title(title)
    ax.set_xlim(0, env.n_cols)
    ax.set_ylim(0, env.n_rows)
    ax.set_xticks(np.arange(env.n_cols + 1))
    ax.set_yticks(np.arange(env.n_rows + 1))
    ax.grid(True)
    ax.invert_yaxis()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Sombrear estados terminais
    for (r, c) in env.terminal_states:
        rect = plt.Rectangle((c, r), 1, 1, fill=True, alpha=0.15)
        ax.add_patch(rect)

    # Valores e setas
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            s = (r, c)
            ax.text(c + 0.5, r + 0.45, value_fmt.format(V[r, c]),
                    ha="center", va="center", fontsize=12)
            if policy is not None:
                a = policy.get(s, "·") or "·"
                ax.text(c + 0.5, r + 0.78, ARROW.get(a, "·"),
                        ha="center", va="center", fontsize=18)

    if created_fig:
        plt.tight_layout()

    return fig


def plot_policy_iteration_history(
    # env: Gridworld,           # A APAGAR
    env: MDP_Gridworld,
    history: list,
) -> plt.Figure:
    """Grelha de subplots com cada iteração externa do policy iteration."""
    num_plots = len(history)
    fig, axes = plt.subplots(1, num_plots, figsize=(num_plots * 6, 6))

    if num_plots == 1:
        axes = [axes]

    for i, (outer_iter, pe_iters, V_hist, pi_actions_hist) in enumerate(history):
        plot_grid(
            env,
            V_hist,
            policy=pi_actions_hist,
            title=(
                f"Policy Iteration (Outer Loop) {outer_iter}\n"
                f"Policy Evaluation (Inner Loop) Itrs: {pe_iters}"
            ),
            ax=axes[i],
        )

    plt.tight_layout()
    return fig


def plot_policy_evaluation_history(
    # env: Gridworld,           # A APAGAR
    env: MDP_Gridworld,
    V_history: List[np.ndarray],
    iters_history: List[int],
    specific_iterations: Optional[List[int]] = None,
) -> None:
    """Plota snapshots de V ao longo da avaliação iterativa de política.

    Args:
        specific_iterations: lista de índices a visualizar (por omissão: [0,1,2,3,4,8,50,100]).
    """
    if specific_iterations is None:
        specific_iterations = [0, 1, 2, 3, 4, 8, 50, 100]

    plot_policy_for_pe = {
        s: "·" if env.is_terminal(s) else "?" for s in env.states()
    }
    indices_to_plot = sorted({it for it in specific_iterations if it < len(V_history)})

    for i in indices_to_plot:
        plot_grid(
            env,
            V_history[i],
            policy=plot_policy_for_pe,
            title=f"Policy Evaluation: V^π (uniform random π) (Iteration {iters_history[i]})",
        )
        plt.show()