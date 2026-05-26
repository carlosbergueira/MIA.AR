"""
Gridworld environment — model-based MDP (determinístico).

Estado   : (row, col)
Acção    : "U", "D", "L", "R"
Dinâmica : determinista; bater na parede → ficar no mesmo sítio.
Recompensa: step_reward em cada passo; 0 nos estados terminais.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


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
# class Gridworld:              A APAGAR
class MDP_Gridworld:
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

    def is_terminal(self, state: Tuple[int, int]) -> bool:
        return state in self.terminal_states

    def step(
        self, state: Tuple[int, int], action: str
    ) -> Tuple[Tuple[int, int], float, bool]:
        """Transição determinista.

        Devolve (next_state, reward, done).
        Se a acção levar para fora da grelha, o agente fica no mesmo sítio.
        """
        if self.is_terminal(state):
            return state, 0.0, True

        d_row, d_col = ACTION_TO_DELTA[action]
        next_row, next_col = state[0] + d_row, state[1] + d_col

        # bater na parede → ficar no sítio
        if next_row < 0 or next_row >= self.n_rows or next_col < 0 or next_col >= self.n_cols:
            next_state = state
        else:
            next_state = (next_row, next_col)

        done = self.is_terminal(next_state)
        return next_state, self.step_reward, done


# class TrapGridworld(Gridworld):               A APAGAR
class TrapGridworld(MDP_Gridworld):
    """Gridworld com célula armadilha (recompensa -10 ao entrar)."""

    trap: Tuple[int, int] = (0, 2)

    def step(
        self, state: Tuple[int, int], action: str
    ) -> Tuple[Tuple[int, int], float, bool]:
        ns, r, done = super().step(state, action)
        if (not self.is_terminal(state)) and ns == self.trap:
            r = -10.0
        return ns, r, done