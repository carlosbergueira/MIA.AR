from __future__ import annotations

import random
from collections import defaultdict

from mia_rl.agents.control.base import ActionT, ControlAgent, StateT
from mia_rl.core.base import Transition


class NStepSarsaControl(ControlAgent[StateT, ActionT]):
    def __init__(
        self,
        actions: tuple[ActionT, ...],
        n_steps: int = 4,
        alpha: float = 0.5,
        epsilon: float = 0.1,
        gamma: float = 1.0,
        seed: int | None = None,
    ):
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1.")

        self.actions = actions
        self.n_steps = n_steps
        self.alpha = alpha
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        super().__init__(gamma=gamma)

    def reset(self) -> None:
        self.Q = defaultdict(float)
        self._selected_actions: dict[StateT, ActionT] = {}
        self._pending_transitions: list[Transition[StateT, ActionT]] = []

    def select_action(self, state: StateT) -> ActionT:
        """Choose an epsilon-greedy action and cache it for the n-step bootstrap.

        TODO:
        1. With probability `self.epsilon`, choose a random action from `self.actions`.
        2. Otherwise choose an action with the highest current action-value.
        3. Store the chosen action in `self._selected_actions[state]` and return it.
        """
        if self.rng.random() < self.epsilon:
            action = self.rng.choice(self.actions)
        else:
            max_value = max(self.Q[(state, a)] for a in self.actions)
            best_actions = [a for a in self.actions if self.Q[(state, a)] == max_value]
            action = self.rng.choice(best_actions)

        self._selected_actions[state] = action
        return action

    def update_transition(self, transition: Transition[StateT, ActionT]) -> None:
        """Store the transition and update the oldest state-action when possible.

        TODO:
        1. Append each transition to `self._pending_transitions`.
        2. If the episode ended, keep updating and removing the oldest transition until the buffer is empty.
        3. Otherwise, once the buffer length reaches `self.n_steps`, update the oldest transition and remove it.
        4. Reuse `_update_oldest_transition()` for the actual target computation.
        """
        # guardar transição
        self._pending_transitions.append(transition)

        if transition.done:
            # no final do episódio esvaziar buffer
            while self._pending_transitions:
                self._update_oldest_transition()
                self._pending_transitions.pop(0)

        else:
            # atualizar quando buffer atingir n passos
            if len(self._pending_transitions) >= self.n_steps:
                self._update_oldest_transition()
                self._pending_transitions.pop(0)

    def _update_oldest_transition(self) -> None:
        """Compute the n-step Sarsa target for the oldest transition in the buffer.

        TODO:
        1. Build a window with at most `self.n_steps` transitions starting from the oldest one.
        2. Sum the discounted rewards inside that window.
        3. If the window has exactly `self.n_steps` transitions and is non-terminal, bootstrap from
            `Q(last_step.next_state, cached_next_action)`.
        4. Apply the incremental update with `self.alpha` to the oldest `(state, action)` pair.
        """
        window = self._pending_transitions[: self.n_steps]

        # soma das recompensas descontadas
        G = 0.0
        for i, transition in enumerate(window):
            G += (self.gamma ** i) * transition.reward

        last_step = window[-1]

        # bootstrap se não terminou e temos n passos completos
        if len(window) == self.n_steps and not last_step.done:
            next_state = last_step.next_state
            next_action = self._selected_actions[next_state]
            G += (self.gamma ** self.n_steps) * self.Q[(next_state, next_action)]

        # transição mais antiga
        first = window[0]
        state = first.state
        action = first.action

        current_q = self.Q[(state, action)]

        # atualização incremental
        self.Q[(state, action)] += self.alpha * (G - current_q)

    def action_value_of(self, state: StateT, action: ActionT) -> float:
        return float(self.Q[(state, action)])

    def greedy_action(self, state: StateT) -> ActionT:
        return max(self.actions, key=lambda action: self.action_value_of(state, action))