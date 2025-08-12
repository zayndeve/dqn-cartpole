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
    Save a line plot showing rewards per episode.

    Args:
        rewards (Sequence[float]): Episode rewards collected during training.
        path (str): Output file path for the PNG plot.
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
    """
    Return the current timestamp formatted as:
    YYYY-MM-DD_HH-MM-SS
    Useful for naming log files or model checkpoints.
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def save_json(obj: dict, path: str) -> None:
    """
    Save a dictionary to a JSON file with pretty printing.

    Args:
        obj (dict): Dictionary to save.
        path (str): Output file path for the JSON file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def set_seed(seed: int, env: Optional[object] = None) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, PyTorch,
    and optionally a Gymnasium environment.

    Args:
        seed (int): Seed value.
        env (Optional[object]): Optional Gymnasium environment to seed.
    """
    # Python's built-in RNG
    random.seed(seed)
    # NumPy RNG
    np.random.seed(seed)
    # PyTorch CPU RNG
    torch.manual_seed(seed)
    # PyTorch GPU RNG (if CUDA is available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Environment-specific seed (if provided)
    if env is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            # Older gym versions may use different reset signatures
            pass
