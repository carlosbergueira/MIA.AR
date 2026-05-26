"""
Funções de visualização para o Gridworld.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from mia_rl.envs.gridworld import ARROW, Gridworld


def plot_grid(
    env: Gridworld,
    V: np.ndarray,
    policy: Optional[Dict[Tuple[int, int], str]] = None,
    title: str = "",
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
            ax.text(c + 0.5, r + 0.45, f"{V[r, c]:.2f}",
                    ha="center", va="center", fontsize=12)
            if policy is not None:
                a = policy.get(s, "·") or "·"
                ax.text(c + 0.5, r + 0.78, ARROW.get(a, "·"),
                        ha="center", va="center", fontsize=18)

    if created_fig:
        plt.tight_layout()

    return fig


def plot_policy_iteration_history(
    env: Gridworld,
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