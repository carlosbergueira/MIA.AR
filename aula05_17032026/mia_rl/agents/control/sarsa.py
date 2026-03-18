from __future__ import annotations

import random
from collections import defaultdict

from mia_rl.agents.control.base import ActionT, ControlAgent, StateT
from mia_rl.core.base import Transition


class SarsaControl(ControlAgent[StateT, ActionT]):
    def __init__(
        self,
        actions: tuple[ActionT, ...],
        alpha: float = 0.5,
        epsilon: float = 0.1,
        gamma: float = 1.0,
        seed: int | None = None,
    ):
        self.actions = actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        super().__init__(gamma=gamma)

    def reset(self) -> None:
        self.Q = defaultdict(float)
        self._selected_actions: dict[StateT, ActionT] = {}

    def select_action(self, state: StateT) -> ActionT:
        """Choose an epsilon-greedy action and cache it for the SARSA bootstrap.

        TODO:
        1. With probability `self.epsilon`, choose a random action from `self.actions`.
        2. Otherwise choose an action with the highest current action-value.
        3. Store the chosen action in `self._selected_actions[state]` and return it.
        """
        # exploração
        if self.rng.random() < self.epsilon:
            action = self.rng.choice(self.actions)
        else:
            # exploração greedy
            max_value = max(self.Q[(state, a)] for a in self.actions)
            best_actions = [a for a in self.actions if self.Q[(state, a)] == max_value]
            action = self.rng.choice(best_actions)

        # guardar ação escolhida
        self._selected_actions[state] = action
        return action

    def update_transition(self, transition: Transition[StateT, ActionT]) -> None:
        """Apply the SARSA update using the cached next action for the next state.

        TODO:
        1. Use a bootstrap value of `0.0` on terminal transitions.
        2. Otherwise read the cached next action from `self._selected_actions[transition.next_state]`.
        3. Compute the SARSA target `reward + gamma * Q(next_state, next_action)`.
        4. Apply the incremental update with `self.alpha`.
        """
        state = transition.state
        action = transition.action
        reward = transition.reward
        next_state = transition.next_state
        done = transition.done

        # valor atual
        current_q = self.Q[(state, action)]

        # bootstrap
        if done:
            target = reward
        else:
            next_action = self._selected_actions[next_state]
            target = reward + self.gamma * self.Q[(next_state, next_action)]

        # atualização incremental
        self.Q[(state, action)] += self.alpha * (target - current_q)

    def action_value_of(self, state: StateT, action: ActionT) -> float:
        return float(self.Q[(state, action)])

    def greedy_action(self, state: StateT) -> ActionT:
        return max(self.actions, key=lambda action: self.action_value_of(state, action))
