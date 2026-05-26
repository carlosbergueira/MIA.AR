"""
Comparação de Monte Carlo, TD(0) e TD(n) no ambiente Blackjack.

Treina os três agentes com a mesma política e número de episódios,
e compara as funções de valor estimadas em checkpoints intermédios.
"""

from __future__ import annotations

from pathlib import Path

from mia_rl.agents.prediction.monte_carlo import FirstVisitMonteCarloPrediction
from mia_rl.agents.prediction.td import TD0Prediction
from mia_rl.agents.prediction.td_n import TDnPrediction
from mia_rl.envs.blackjack import BlackjackEnv
from mia_rl.experiments.training import train_prediction_agent
from mia_rl.plots.blackjack import plot_value_function, plot_value_difference
from mia_rl.policies.blackjack import ThresholdPolicy


def run_tdn_comparison(
    output_dir: Path,
    n: int = 5,
    num_episodes: int = 500_000,
    alpha: float = 0.05,
    gamma: float = 1.0,
    seed: int = 42,
) -> None:
    """
    Treina FirstVisitMC, TD(0) e TD(n) no Blackjack com a mesma política fixa
    e guarda gráficos comparativos das funções de valor.

    Args:
        output_dir:   pasta onde guardar os gráficos.
        n:            número de passos para o TD(n).
        num_episodes: número de episódios de treino.
        alpha:        learning rate para TD(0) e TD(n).
        gamma:        fator de desconto.
        seed:         semente para reproducibilidade.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = [1_000, 10_000, 50_000, num_episodes]
    policy = ThresholdPolicy(threshold=17)

    # ── Instanciar agentes ──────────────────────────────────────────────────
    agents = {
        "First-Visit MC":  FirstVisitMonteCarloPrediction(gamma=gamma),
        "TD(0)":           TD0Prediction(alpha=alpha, gamma=gamma),
        f"TD({n})":        TDnPrediction(n=n, alpha=alpha, gamma=gamma),
    }

    histories: dict[str, dict] = {}

    # ── Treinar cada agente ─────────────────────────────────────────────────
    for name, agent in agents.items():
        print(f"\nA treinar {name}...")
        env = BlackjackEnv(seed=seed)
        history = train_prediction_agent(
            env=env,
            policy=policy,
            agent=agent,
            num_episodes=num_episodes,
            checkpoints=checkpoints,
        )
        histories[name] = history
        print(f"  Concluído ({num_episodes} episódios).")

    # ── Gráficos das funções de valor no final do treino ────────────────────
    print("\nA gerar gráficos das funções de valor...")
    for name, history in histories.items():
        values = history[num_episodes]
        fig, _ = plot_value_function(values, title=f"V estimado — {name} ({num_episodes} ep)")
        fname = f"value_{name.replace('(', '').replace(')', '').replace(' ', '_').lower()}.png"
        fig.savefig(output_dir / fname)
        print(f"  Guardado: {output_dir / fname}")

    # ── Diferenças entre agentes (MC como referência) ───────────────────────
    print("\nA gerar gráficos de diferença relativamente a MC...")
    mc_values = histories["First-Visit MC"][num_episodes]

    for name in [f"TD({n})", "TD(0)"]:
        other_values = histories[name][num_episodes]
        fig, _ = plot_value_difference(
            mc_values,
            other_values,
            title=f"Diferença: First-Visit MC − {name}",
        )
        fname = f"diff_mc_vs_{name.replace('(', '').replace(')', '').replace(' ', '_').lower()}.png"
        fig.savefig(output_dir / fname)
        print(f"  Guardado: {output_dir / fname}")

    # ── Evolução ao longo dos checkpoints (sem usable ace, dealer=1) ────────
    print("\nA gerar gráfico de convergência por checkpoint...")
    _plot_convergence(histories, checkpoints, output_dir)

    print("\nConcluído. Resultados em:", output_dir)


# ── Helper: convergência ────────────────────────────────────────────────────

def _plot_convergence(
    histories: dict[str, dict],
    checkpoints: list[int],
    output_dir: Path,
) -> None:
    """
    Para um estado representativo (player_sum=20, dealer=1, no usable ace),
    mostra como V(s) evolui ao longo dos checkpoints para os três agentes.
    """
    import matplotlib.pyplot as plt

    probe_state = (20, 1, False)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for name, history in histories.items():
        values_at_checkpoints = [
            history[cp].get(probe_state, 0.0) for cp in checkpoints
        ]
        ax.plot(checkpoints, values_at_checkpoints, marker="o", label=name)

    ax.set_xscale("log")
    ax.set_xlabel("Episódios (escala log)")
    ax.set_ylabel(f"V{probe_state}")
    ax.set_title(f"Convergência de V{probe_state} por agente")
    ax.legend()

    fname = "convergence_probe_state.png"
    fig.savefig(output_dir / fname)
    plt.close(fig)
    print(f"  Guardado: {output_dir / fname}")
