from __future__ import annotations

from collections import defaultdict

from mia_rl.core.base import Episode, PredictionAgent
from mia_rl.envs.blackjack import BlackjackAction, BlackjackState


class TDnPrediction(PredictionAgent[BlackjackState, BlackjackAction]):
    """
    N-step TD Prediction.

    Generaliza TD(0) (n=1) e First-Visit Monte Carlo (n=infinito).
    Para cada estado no passo t, o target é calculado com as próximas
    n recompensas reais seguidas de bootstrap com V(s_{t+n}):

        G(n)_t = r_{t+1} + γr_{t+2} + ... + γ^{n-1}r_{t+n} + γ^n * V(s_{t+n})

    Se o episódio terminar antes de n passos, não há bootstrap (como MC).
    """

    def __init__(self, n: int = 1, alpha: float = 0.05, gamma: float = 1.0):
        self.n = n
        self.alpha = alpha
        super().__init__(gamma=gamma)

    def reset(self) -> None:
        self.V: dict[BlackjackState, float] = defaultdict(float)

    def update_episode(self, episode: Episode[BlackjackState, BlackjackAction]) -> None:
        T = len(episode.transitions)

        for t in range(T):
            # Acumular as próximas n recompensas descontadas a partir de t
            G = 0.0
            for i in range(self.n):
                step = t + i
                if step >= T:
                    break
                G += (self.gamma ** i) * episode.transitions[step].reward

            # Bootstrap com V(s_{t+n}) se ainda não chegámos ao fim
            bootstrap_step = t + self.n
            if bootstrap_step < T:
                bootstrap_state = episode.transitions[bootstrap_step].state
                G += (self.gamma ** self.n) * self.V[bootstrap_state]

            # Atualizar V(s_t)
            state = episode.transitions[t].state
            self.V[state] += self.alpha * (G - self.V[state])

    def value_of(self, state: BlackjackState) -> float:
        return float(self.V[state])
