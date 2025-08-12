# dqn.py
# --------------------------------------------
# Plain DQN (optionally Dueling) for CartPole.
# PyTorch, simple MLP. Compatible with train.py.
# --------------------------------------------
from __future__ import annotations
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


def _init_weights(m: nn.Module) -> None:
    """Kaiming/Xavier initialization for linear layers to improve stability."""
    if isinstance(m, nn.Linear):
        # He/Kaiming uniform works well with ReLU activations
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class DQN(nn.Module):
    """
    DQN network for low-dimensional state spaces (e.g., CartPole).

    If enable_dueling_dqn=False → classic DQN head (as in 2013 DQN, but MLP).
    If enable_dueling_dqn=True  → dueling head (Value + Advantage streams).

    Args:
        state_dim:  dimension of observation (CartPole: 4)
        action_dim: number of discrete actions (CartPole: 2)
        hidden_dim: width of hidden layers
        enable_dueling_dqn: toggles dueling architecture (OFF for professor's spec)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        enable_dueling_dqn: bool = False,
    ) -> None:
        super().__init__()
        self.enable_dueling_dqn = enable_dueling_dqn

        # Two-layer MLP backbone (more stable than single layer on CartPole)
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        if self.enable_dueling_dqn:
            # Dueling streams
            self.fc_value = nn.Linear(hidden_dim, hidden_dim)
            self.value = nn.Linear(hidden_dim, 1)

            self.fc_adv = nn.Linear(hidden_dim, hidden_dim)
            self.adv = nn.Linear(hidden_dim, action_dim)
        else:
            # Classic DQN head
            self.out = nn.Linear(hidden_dim, action_dim)

        # Initialize
        self.apply(_init_weights)

        # Small tweak: output layer bias to zero helps early training stability
        if not self.enable_dueling_dqn:
            nn.init.zeros_(self.out.bias)
        else:
            nn.init.zeros_(self.value.bias)
            nn.init.zeros_(self.adv.bias)

        # Expose for convenience (used once in some codebases)
        self.output_dim: int = action_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MLP trunk
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        if self.enable_dueling_dqn:
            # Value branch
            v = F.relu(self.fc_value(x))
            V = self.value(v)  # [B, 1]

            # Advantage branch
            a = F.relu(self.fc_adv(x))
            A = self.adv(a)    # [B, A]

            # Combine into Q-values:
            # Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
            Q = V + A - A.mean(dim=1, keepdim=True)
        else:
            Q = self.out(x)

        return Q


if __name__ == "__main__":
    # quick smoke test
    state_dim, action_dim = 4, 2
    net = DQN(state_dim, action_dim, hidden_dim=128, enable_dueling_dqn=False)
    x = torch.randn(10, state_dim)
    y = net(x)
    print("Output shape:", y.shape)  # (10, 2)
