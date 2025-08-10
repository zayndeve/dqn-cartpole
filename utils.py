import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

def save_plot(rewards, path="saved_models/learning_curve.png"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward per Episode")
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
