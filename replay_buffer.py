# replay_buffer.py
# ----------------------------------------------------
# Simple uniform replay buffer (as in original DQN).
# Compatible with train.py: .append(tuple) and .sample(N)
# Also includes ExperienceReplay.push(...) as an alias.
# ----------------------------------------------------
from __future__ import annotations
from typing import Deque, List, Tuple, Optional
from collections import deque
import random


Transition = Tuple  # (state, action, reward, next_state, done)


class ReplayMemory:
    """
    Uniform experience replay buffer.

    Args:
        maxlen: maximum number of transitions to keep
        seed:   optional Python RNG seed for reproducibility
    """

    def __init__(self, maxlen: int, seed: Optional[int] = None) -> None:
        self.memory: Deque[Transition] = deque(maxlen=maxlen)
        if seed is not None:
            random.seed(seed)

    def append(self, transition: Transition) -> None:
        """Add one transition: (s, a, r, s2, done)."""
        self.memory.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        """Uniformly sample a batch of transitions."""
        # random.sample raises ValueError if not enough elements; caller checks len()
        return random.sample(self.memory, batch_size)

    def clear(self) -> None:
        """Remove all stored transitions."""
        self.memory.clear()

    def __len__(self) -> int:
        return len(self.memory)


# Backward/compatibility alias
class ExperienceReplay(ReplayMemory):
    """
    Drop-in alias with a 'push' method, used by some older code.
    """
    def push(self, state, action, reward, next_state, done) -> None:
        self.append((state, action, reward, next_state, done))
