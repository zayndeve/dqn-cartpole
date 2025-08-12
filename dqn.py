import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Simple 2-layer MLP for classic-control tasks.
    - If enable_dueling_dqn is False → standard DQN head.
    - If True → dueling heads (value + advantages).
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, enable_dueling_dqn=False):
        super().__init__()
        self.enable_dueling_dqn = enable_dueling_dqn

        # shared feature extractor (2 layers)
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        if self.enable_dueling_dqn:
            # value stream
            self.fc_value = nn.Linear(hidden_dim, hidden_dim)
            self.value = nn.Linear(hidden_dim, 1)
            # advantage stream
            self.fc_advantages = nn.Linear(hidden_dim, hidden_dim)
            self.advantages = nn.Linear(hidden_dim, action_dim)
        else:
            self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        if self.enable_dueling_dqn:
            v = F.relu(self.fc_value(x))
            V = self.value(v)
            a = F.relu(self.fc_advantages(x))
            A = self.advantages(a)
            Q = V + A - A.mean(dim=1, keepdim=True)
        else:
            Q = self.output(x)
        return Q


if __name__ == "__main__":
    s, a = 4, 2
    net = DQN(s, a, hidden_dim=256, enable_dueling_dqn=False)
    print(net(torch.randn(3, s)).shape)  # -> (3, 2)
