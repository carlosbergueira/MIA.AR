from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def run_car_rental(output_dir, gamma=0.9):
    from mia_rl.envs.car_rental import CarRentalMDP, CarRentalParams
    from mia_rl.agents.planning.car_rental import policy_iteration, value_iteration
    from mia_rl.plots.car_rental import plot_policy, plot_values

    params = CarRentalParams()
    mdp = CarRentalMDP(params)

    # Policy Iteration
    V_pi, policy_pi, history = policy_iteration(mdp, gamma=gamma)

    fig1 = plot_policy(mdp, policy_pi, title="Policy Iteration")
    fig1.savefig(output_dir / "policy_iteration_policy.png")
    print(f"Saved: {output_dir / 'policy_iteration_policy.png'}")

    fig2 = plot_values(mdp, V_pi, title="Values (PI)")
    fig2.savefig(output_dir / "policy_iteration_values.png")
    print(f"Saved: {output_dir / 'policy_iteration_values.png'}")

    # Value Iteration
    V_vi, policy_vi, iters_vi = value_iteration(mdp, gamma=gamma)

    fig3 = plot_policy(mdp, policy_vi, title="Value Iteration")
    fig3.savefig(output_dir / "value_iteration_policy.png")
    print(f"Saved: {output_dir / 'value_iteration_policy.png'}")

    fig4 = plot_values(mdp, V_vi, title="Values (VI)")
    fig4.savefig(output_dir / "value_iteration_values.png")
    print(f"Saved: {output_dir / 'value_iteration_values.png'}")