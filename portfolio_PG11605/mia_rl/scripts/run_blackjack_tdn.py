"""
Script executável para comparar First-Visit MC, TD(0) e TD(n) no Blackjack.

Uso:
    python mia_rl/scripts/run_blackjack_tdn.py
    python mia_rl/scripts/run_blackjack_tdn.py --n 3 --episodes 200000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Garantir que o repo root está no PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPO_ROOT = Path(__file__).resolve().parents[2]
# PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mia_rl.experiments.blackjack import run_tdn_comparison

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PACKAGE_ROOT / "outputs" / "blackjack_tdn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comparação MC vs TD(0) vs TD(n) no Blackjack")
    parser.add_argument("--n",        type=int,   default=5,       help="Número de passos para TD(n) (default: 5)")
    parser.add_argument("--episodes", type=int,   default=500_000, help="Número de episódios de treino (default: 500000)")
    parser.add_argument("--alpha",    type=float, default=0.05,    help="Learning rate para TD (default: 0.05)")
    parser.add_argument("--gamma",    type=float, default=1.0,     help="Fator de desconto (default: 1.0)")
    parser.add_argument("--seed",     type=int,   default=42,      help="Semente aleatória (default: 42)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Configuração: n={args.n}, episódios={args.episodes}, alpha={args.alpha}, gamma={args.gamma}")
    print(f"Output: {OUTPUT_DIR}\n")

    run_tdn_comparison(
        output_dir=OUTPUT_DIR,
        n=args.n,
        num_episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        seed=args.seed,
    )
