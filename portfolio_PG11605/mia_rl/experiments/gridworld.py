from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def run_gridworld(output_dir: Path, gamma: float = 0.9) -> None:
    from mia_rl.envs.gridworld import Gridworld
    from mia_rl.agents.planning.gridworld import (
        uniform_random_policy,
        policy_evaluation,
        policy_improvement,
        policy_iteration,
    )
    from mia_rl.plots.gridworld import plot_grid, plot_policy_iteration_history

    env = Gridworld()

    # ---- Avaliação da política uniforme aleatória ----
    pi0 = uniform_random_policy(env)
    V_pi0, iters = policy_evaluation(env, pi0, gamma=gamma)
    print(f"Policy evaluation (uniform random) converged in {iters} iterations.")

    fig = plot_grid(env, V_pi0, title="Policy Evaluation: V^π (uniform random π)")
    fig.savefig(output_dir / "policy_evaluation_uniform.png")
    print(f"Saved: {output_dir / 'policy_evaluation_uniform.png'}")

    # ---- Política gulosa a partir de V^π0 ----
    pi1_actions, _ = policy_improvement(env, V_pi0, old_policy_actions=None, gamma=gamma)
    fig = plot_grid(env, V_pi0, policy=pi1_actions,
                    title="Greedy policy w.r.t. V^π (arrows)")
    fig.savefig(output_dir / "policy_improvement_greedy.png")
    print(f"Saved: {output_dir / 'policy_improvement_greedy.png'}")

    # ---- Iteração de política completa ----
    V_star, pi_star_actions, history = policy_iteration(env, gamma=gamma)
    print(f"Policy iteration outer loops: {len(history)}")

    fig = plot_grid(env, V_star, policy=pi_star_actions,
                    title="Policy Iteration: V* and π* (greedy actions)")
    fig.savefig(output_dir / "policy_iteration_final.png")
    print(f"Saved: {output_dir / 'policy_iteration_final.png'}")

    # ---- Histórico de iterações ----
    fig = plot_policy_iteration_history(env, history)
    fig.savefig(output_dir / "policy_iteration_history.png")
    print(f"Saved: {output_dir / 'policy_iteration_history.png'}")