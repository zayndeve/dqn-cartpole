# DQN CartPole Project

Implementation of a **Deep Q-Network (DQN)** agent to solve the `CartPole-v1` environment from [Gymnasium](https://gymnasium.farama.org/) using [PyTorch](https://pytorch.org/).

## 📌 Project Overview

This project was developed as part of the **Engineering Design** course under the supervision of Prof. Han Youn-Hee.  
The goal is to train an AI agent to balance a pole on a cart for as long as possible, achieving the maximum reward of **500**.

## 🎯 Objectives

- Implement the DQN algorithm described in [Mnih et al., 2013](https://arxiv.org/abs/1312.5602).
- Use PyTorch as the deep learning framework.
- Achieve an Episode Reward of **500**.
- Document results and code in a Notion page.

## 🛠 Tech Stack

- **Language:** Python 3.11
- **Libraries:** PyTorch, Gymnasium, NumPy, Matplotlib, tqdm, OpenCV, MoviePy

## 📂 Project Structure

```
DQN-CartPole/
├── .venv/               # Virtual environment (not pushed to GitHub)
├── cartpole_test.py     # Test file to check environment setup
├── README.md            # Project documentation
└── .gitignore           # Ignore venv & cache files
```

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<YOUR_USERNAME>/dqn-cartpole.git
   cd dqn-cartpole
   ```
2. Create and activate virtual environment:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run test file:
   ```bash
   python cartpole_test.py
   ```

## 📊 Results

- Target: **Episode Reward = 500** (to be updated after training)
- Screenshots and gameplay video will be added here.

---

**Author:** Ziynatilloh Tursunboev  
**Supervisor:** Prof. Han Youn-Hee
