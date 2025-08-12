# replay_buffer.py
# ----------------------------------------------------
# Simple uniform replay buffer (as in the original DQN paper).
# Compatible with train.py:
#   - append(tuple): add a transition
#   - sample(N):     retrieve a random batch of transitions
# Includes ExperienceReplay.push(...) as a backward-compatible alias.
# ----------------------------------------------------
from __future__ import annotations
from typing import Deque, List, Tuple, Optional
from collections import deque
import random

# Transition tuple format: (state, action, reward, next_state, done)
Transition = Tuple


class ReplayMemory:
    """
    Uniform experience replay buffer.

    Stores past transitions up to a fixed maximum size. When full,
    older transitions are automatically discarded (FIFO).

    Args:
        maxlen (int): Maximum number of transitions to keep.
        seed (Optional[int]): Optional seed for Python's random module
                              (useful for reproducibility).
    """

    def __init__(self, maxlen: int, seed: Optional[int] = None) -> None:
        # Use deque for efficient append and pop from both ends
        self.memory: Deque[Transition] = deque(maxlen=maxlen)
        if seed is not None:
            random.seed(seed)

    def append(self, transition: Transition) -> None:
        """
        Add a single transition to the replay buffer.
        
        Args:
            transition (tuple): (state, action, reward, next_state, done)
        """
        self.memory.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        """
        Uniformly sample a batch of transitions without replacement.

        Args:
            batch_size (int): Number of samples to return.
        
        Returns:
            List[Transition]: Randomly sampled transitions.
        
        Raises:
            ValueError: If batch_size > number of stored transitions.
                        (Caller should check len(self) first.)
        """
        return random.sample(self.memory, batch_size)

    def clear(self) -> None:
        """Remove all stored transitions from the buffer."""
        self.memory.clear()

    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return len(self.memory)


# Backward compatibility alias for older codebases
class ExperienceReplay(ReplayMemory):
    """
    Drop-in alias for ReplayMemory with a 'push' method
    (older code may expect push(...) instead of append(...)).
    """
    def push(self, state, action, reward, next_state, done) -> None:
        """Alias to append() for backward compatibility."""
        self.append((state, action, reward, next_state, done))
