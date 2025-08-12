# DQN CartPole Project

Implementation of a Deep Q-Network (DQN) agent to solve the `CartPole-v1` environment from [Gymnasium](https://gymnasium.farama.org/) using [PyTorch](https://pytorch.org/).

This project was developed as part of the Engineering Design course under the supervision of Prof. Han Youn-Hee.

## Overview

The CartPole problem is a classic reinforcement learning control task where an agent must balance a pole upright on a moving cart by applying forces to the left or right. The goal is to keep the pole balanced for 500 timesteps (the maximum allowed), achieving the maximum episode reward.

This implementation follows the algorithm described in:

Mnih et al., 2013 – Playing Atari with Deep Reinforcement Learning  
https://arxiv.org/abs/1312.5602

Key features:
- Experience replay buffer
- Target network for stable Q-learning
- Epsilon-greedy action selection
- Optional Dueling DQN architecture

## Objectives

1. Implement a functional DQN agent in PyTorch.
2. Train the agent to achieve an episode reward of 500.
3. Record training metrics and videos.
4. Produce reproducible results and organized experiment outputs.

## Technology Stack

- Python 3.11
- PyTorch ≥ 2.1
- Gymnasium 0.29.1
- NumPy ≥ 1.23
- Matplotlib ≥ 3.7

## Project Structure

```
DQN-CartPole/
├── dqn.py               # DQN model architecture
├── train.py             # Agent, training loop, optimization
├── eval.py              # Evaluation and video recording
├── replay_buffer.py     # Experience replay buffer
├── utils.py             # Utilities for plotting, seeding, etc.
├── hyperparameters.yml  # Training configuration profiles
├── requirements.txt     # Python dependencies
├── saved_models/        # Trained model weights
├── videos/              # Recorded videos
├── results/             # Organized run outputs
└── README.md            # Project documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<YOUR_USERNAME>/DQN-CartPole.git
   cd DQN-CartPole
   ```

2. Create and activate a virtual environment:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training

To start training using a profile from `hyperparameters.yml`:

```bash
python train.py cartpole1 --episodes 500
```

Outputs:
- `saved_models/best.pt` – best model weights
- `saved_models/last.pt` – final model weights
- `saved_models/learning_curve.png` – reward per episode plot

## Evaluation

Evaluate the best model with a real-time render:

```bash
python eval.py cartpole1 --human --model saved_models/best.pt
```

Record evaluation as video:

```bash
python eval.py cartpole1 --video --model saved_models/best.pt
```

## Results & Media

All results for each training run are stored under `results/<timestamp>/`:

- `train.log` – full training log
- `best.pt` – best model weights
- `last.pt` – final model weights
- `learning_curve.png` – training plot
- `videos/` – recorded evaluation videos
- `hyperparameters.yml` – config used for the run
- `requirements.txt` – dependency snapshot
- `git_commit.txt` – repository commit hash (if applicable)

Example to open results for a specific run:
```
results/20250812-153000/learning_curve.png
results/20250812-153000/videos/cartpole-episode-1.mp4
```

### Example Training Output

#### Learning Curve
Reward progression per episode during training:
![Learning Curve]
<img width="571" height="455" alt="learning_curve" src="https://github.com/user-attachments/assets/2ed0932b-f2f0-4fae-8ab0-fd79e96e545b" />



#### Training Log (Excerpt)
Example console log output:
```
[Episode    1] reward= 12.0  eps=1.000
[Episode    2] reward= 25.0  eps=0.995
...
[Episode   88] reward=500.0  eps=0.051
Reached target reward. Stopping training.
```
*(You can also use a screenshot of your terminal output here.)*

#### Evaluation Video
Example of the trained agent balancing the pole:

![Evaluation Video]
https://github.com/user-attachments/assets/108850e2-8500-4ae6-a58a-84d5a2a94a47



## Reproducibility

To reset all outputs and start fresh:

```bash
rm -rf saved_models/* videos/*
python train.py cartpole1 --episodes 500
```

To automatically archive results for reporting:

```bash
bash scripts/reset_and_run.sh cartpole1 500
```

## File Explanations

- `dqn.py` – Defines the DQN neural network (standard or dueling).
- `train.py` – Agent class, training loop, epsilon-greedy policy.
- `eval.py` – Loads a model, runs episodes, records video.
- `replay_buffer.py` – Stores and samples past experiences.
- `utils.py` – Plotting, seeding, and helper utilities.
- `hyperparameters.yml` – Training profiles with adjustable parameters.

## References

1. Mnih, Volodymyr, et al. "Playing Atari with deep reinforcement learning." arXiv preprint arXiv:1312.5602 (2013).
2. Gymnasium CartPole-v1 Documentation – https://gymnasium.farama.org/environments/classic_control/cart_pole/
3. PyTorch Documentation – https://pytorch.org/docs/stable/index.html

**Author:** Ziynatilloh Tursunboev  
**Supervisor:** Prof. Han Youn-Hee
