# utils.py
# ---------------------------------------------------------
# Utility functions for plotting, logging, and reproducibility.
# ---------------------------------------------------------
import os
import random
import json
from datetime import datetime
from typing import Sequence, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch


def save_plot(rewards: Sequence[float], path: str = "saved_models/learning_curve.png") -> None:
    """
    Save a learning curve plot of rewards per episode.

    Args:
        rewards: sequence of episode rewards
        path: output PNG file path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure()
    plt.plot(rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward per Episode")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def timestamp() -> str:
    """Return current timestamp string (YYYY-MM-DD_HH-MM-SS)."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def save_json(obj: dict, path: str) -> None:
    """Save a dictionary as pretty-printed JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def set_seed(seed: int, env: Optional[object] = None) -> None:
    """
    Set random seed for Python, NumPy, PyTorch, and optionally a Gymnasium env.

    Args:
        seed: integer seed
        env: optional gym.Env or gymnasium.Env to seed directly
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if env is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            pass  # older gym versions may differ
