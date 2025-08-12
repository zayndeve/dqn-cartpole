import os
import argparse
import random
from dataclasses import dataclass

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from dqn import DQN                      # <- renamed from network.py
from replay_buffer import ReplayMemory
from utils import save_plot              # curve png


# -------------------------------
# Config
# -------------------------------
@dataclass
class Config:
    env_id: str = "CartPole-v1"
    replay_memory_size: int = 50_000
    mini_batch_size: int = 64
    epsilon_init: float = 1.0
    epsilon_decay: float = 0.995       # multiplicative per step
    epsilon_min: float = 0.05
    network_sync_rate: int = 200       # hard update every N steps
    learning_rate_a: float = 5e-4
    discount_factor_g: float = 0.99
    stop_on_reward: int = 500
    fc1_nodes: int = 128               # hidden_dim for MLP

def load_config(yaml_path: str, profile: str) -> Config:
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    d = raw.get(profile)
    if d is None:
        raise ValueError(f"Profile '{profile}' not found in {yaml_path}")
    known = {k: d[k] for k in [
        "env_id","replay_memory_size","mini_batch_size","epsilon_init",
        "epsilon_decay","epsilon_min","network_sync_rate","learning_rate_a",
        "discount_factor_g","stop_on_reward","fc1_nodes"
    ] if k in d}
    return Config(**known)


# -------------------------------
# Agent (moved here from agent.py)
# -------------------------------
class Agent:
    def __init__(self, cfg: Config, seed: int = 0, device: str | None = None):
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # envs
        self.env = gym.make(cfg.env_id)
        self.eval_env = gym.make(cfg.env_id)  # no "human" by default; keep eval headless

        # seeding
        np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)

        obs, _ = self.env.reset(seed=seed)
        self.state_dim = int(obs.shape[0])
        self.action_dim = int(self.env.action_space.n)

        # networks (vanilla DQN)
        self.policy_net = DQN(
            self.state_dim, self.action_dim,
            hidden_dim=self.cfg.fc1_nodes,
            enable_dueling_dqn=False
        ).to(self.device)
        self.target_net = DQN(
            self.state_dim, self.action_dim,
            hidden_dim=self.cfg.fc1_nodes,
            enable_dueling_dqn=False
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # optimizer & loss (Huber is a bit stabler than MSE)
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=cfg.learning_rate_a)
        self.criterion = nn.SmoothL1Loss()

        # replay
        self.memory = ReplayMemory(cfg.replay_memory_size)

        # epsilon-greedy
        self.epsilon = cfg.epsilon_init
        self.steps_done = 0

        # io
        os.makedirs("saved_models", exist_ok=True)

    def select_action(self, state: np.ndarray) -> int:
        self.steps_done += 1
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy_net(s)
            return int(q.argmax(dim=1).item())

    def optimize(self):
        if len(self.memory) < self.cfg.mini_batch_size:
            return 0.0

        batch = self.memory.sample(self.cfg.mini_batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        states      = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions     = torch.tensor(actions, dtype=torch.int64,  device=self.device).unsqueeze(1)
        rewards     = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones       = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s,a)
        q_sa = self.policy_net(states).gather(1, actions)

        # target = r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            q_next = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target = rewards + (1.0 - dones) * self.cfg.discount_factor_g * q_next

        loss = self.criterion(q_sa, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # periodic hard update
        if self.steps_done % self.cfg.network_sync_rate == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def train(self, episodes: int = 500):
        best_reward = -float("inf")
        rewards = []

        for ep in range(1, episodes + 1):
            state, _ = self.env.reset()
            ep_reward, done = 0.0, False

            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                ep_reward += reward

                _ = self.optimize()

                # ε decay (clamped)
                if self.epsilon > self.cfg.epsilon_min:
                    self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)

            rewards.append(ep_reward)
            if ep_reward > best_reward:
                best_reward = ep_reward
                self.save("saved_models/best.pt")

            print(f"[Episode {ep:4d}] reward={ep_reward:5.1f}  eps={self.epsilon:.3f}")

            if ep_reward >= self.cfg.stop_on_reward:
                print("Reached target reward. Stopping training.")
                break

        self.save("saved_models/last.pt")
        return rewards

    def evaluate(self, episodes: int = 5, model_path: str | None = None):
        if model_path:
            self.load(model_path)

        env = self.eval_env
        scores = []
        for ep in range(episodes):
            state, _ = env.reset()
            done, ep_reward = False, 0.0
            while not done:
                with torch.no_grad():
                    s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                    action = int(self.policy_net(s).argmax(dim=1).item())  # greedy
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_reward += reward
            scores.append(ep_reward)
            print(f"[Eval {ep+1}/{episodes}] reward={ep_reward:.1f}")
        return scores

    def save(self, path: str):
        torch.save({"model_state_dict": self.policy_net.state_dict(),
                    "cfg": self.cfg.__dict__}, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(data["model_state_dict"])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.eval()


# -------------------------------
# CLI (kept from your old train.py)
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", help="config profile in hyperparameters.yml, e.g., cartpole1")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--yaml", default="hyperparameters.yml")
    parser.add_argument("--outdir", default="saved_models", help="where to save PNG/logs")
    args = parser.parse_args()

    cfg = load_config(args.yaml, args.profile)
    agent = Agent(cfg)

    os.makedirs(args.outdir, exist_ok=True)
    rewards = agent.train(episodes=args.episodes)

    png_path = os.path.join(args.outdir, "learning_curve.png")
    save_plot(rewards, png_path)
    print(f"[OK] Saved learning curve → {png_path}")
    print(f"[INFO] Weights are in {args.outdir}/best.pt and {args.outdir}/last.pt")

if __name__ == "__main__":
    main()
