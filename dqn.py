import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Simple 2-layer MLP for classic-control tasks.
    - If enable_dueling_dqn is False → standard DQN head (outputs Q-values directly).
    - If True → dueling DQN architecture with separate value and advantage streams.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, enable_dueling_dqn=False):
        super().__init__()
        self.enable_dueling_dqn = enable_dueling_dqn

        # Shared feature extractor (two fully connected layers)
        self.fc1 = nn.Linear(state_dim, hidden_dim)   # Input → first hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # First hidden → second hidden layer

        if self.enable_dueling_dqn:
            # Value stream: learns the value of being in a given state
            self.fc_value = nn.Linear(hidden_dim, hidden_dim)
            self.value = nn.Linear(hidden_dim, 1)  # Outputs a single state value V(s)

            # Advantage stream: learns the relative advantage of each action
            self.fc_advantages = nn.Linear(hidden_dim, hidden_dim)
            self.advantages = nn.Linear(hidden_dim, action_dim)  # Outputs advantage A(s, a) for each action
        else:
            # Standard DQN output: directly predicts Q-values for each action
            self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """
        Forward pass:
        Processes the input state through the network and returns Q-values.
        """
        # Pass input through shared feature layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        if self.enable_dueling_dqn:
            # Value branch
            v = F.relu(self.fc_value(x))
            V = self.value(v)

            # Advantage branch
            a = F.relu(self.fc_advantages(x))
            A = self.advantages(a)

            # Combine value and advantages into Q-values
            # Subtract mean advantage to normalize (dueling DQN trick)
            Q = V + A - A.mean(dim=1, keepdim=True)
        else:
            # Standard DQN Q-values
            Q = self.output(x)

        return Q


if __name__ == "__main__":
    # Example usage:
    s, a = 4, 2  # State dimension = 4, Action dimension = 2
    net = DQN(s, a, hidden_dim=256, enable_dueling_dqn=False)
    # Forward pass with random input (batch of 3 states)
    print(net(torch.randn(3, s)).shape)  # Expected output shape: (3, 2)
