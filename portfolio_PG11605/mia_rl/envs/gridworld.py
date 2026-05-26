"""
Gridworld environment — model-based MDP (determinístico).

Estado   : (row, col)
Acção    : "U", "D", "L", "R"
Dinâmica : determinista; bater na parede → ficar no mesmo sítio.
Recompensa: step_reward em cada passo; 0 nos estados terminais.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


ACTIONS = ["U", "D", "L", "R"]

ACTION_TO_DELTA: Dict[str, Tuple[int, int]] = {
    "U": (-1,  0),
    "D": ( 1,  0),
    "L": ( 0, -1),
    "R": ( 0,  1),
}

# CLAUDE deu este: ARROW: Dict[str, str] = {"U": "↑", "D": "↓", "L": "←", "R": "→", "·": "·"}
ARROW = {"U":"↑", "D":"↓", "L":"←", "R":"→", "·":"·"}

@dataclass(frozen=True)
class Gridworld:
    """Grelha rectangular com estados terminais nos cantos (por omissão)."""

    n_rows: int = 4
    n_cols: int = 4
    terminal_states: Tuple[Tuple[int, int], ...] = ((0, 0), (3, 3))
    step_reward: float = -1.0

    # ------------------------------------------------------------------
    # Interface do MDP
    # ------------------------------------------------------------------

    def states(self) -> List[Tuple[int, int]]:
        """Todos os estados (r, c)."""
        return [(r, c) for r in range(self.n_rows) for c in range(self.n_cols)]

    def is_terminal(self, s: Tuple[int, int]) -> bool:
        return s in self.terminal_states

    def step(
        self, s: Tuple[int, int], a: str
    ) -> Tuple[Tuple[int, int], float, bool]:
        """Transição determinista.

        Devolve (next_state, reward, done).
        Se a acção levar para fora da grelha, o agente fica no mesmo sítio.
        """
        if self.is_terminal(s):
            return s, 0.0, True

        dr, dc = ACTION_TO_DELTA[a]
        nr, nc = s[0] + dr, s[1] + dc

        # bater na parede → ficar no sítio
        if nr < 0 or nr >= self.n_rows or nc < 0 or nc >= self.n_cols:
            ns = s
        else:
            ns = (nr, nc)

        done = self.is_terminal(ns)
        return ns, self.step_reward, done