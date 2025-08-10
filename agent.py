import os
import yaml
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from network import DQN
from replay_buffer import ReplayMemory


@dataclass
class Config:
    env_id: str = "CartPole-v1"
    replay_memory_size: int = 50000
    mini_batch_size: int = 64
    epsilon_init: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.05
    network_sync_rate: int = 200
    learning_rate_a: float = 5e-4
    discount_factor_g: float = 0.99
    stop_on_reward: int = 500
    fc1_nodes: int = 128  # will be used as hidden_dim


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


class Agent:
    def __init__(self, cfg: Config, seed: int = 0, device: str | None = None):
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.env = gym.make(cfg.env_id)
        self.eval_env = gym.make(cfg.env_id, render_mode="human")

        # seeding
        np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)

        obs, _ = self.env.reset(seed=seed)
        self.state_dim = int(obs.shape[0])
        self.action_dim = int(self.env.action_space.n)

        # networks (vanilla DQN: dueling disabled)
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

        # optimizer & loss
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=cfg.learning_rate_a)
        self.criterion = nn.MSELoss()

        # replay
        self.memory = ReplayMemory(cfg.replay_memory_size)

        # epsilon-greedy
        self.epsilon = cfg.epsilon_init
        self.steps_done = 0

        # logging
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

        # ReplayMemory.sample returns a list of transitions -> unpack to arrays
        batch = self.memory.sample(self.cfg.mini_batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        states      = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions     = torch.tensor(actions, dtype=torch.int64,  device=self.device).unsqueeze(1)
        rewards     = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones       = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s,a)
        q_values = self.policy_net(states).gather(1, actions)

        # target = r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target = rewards + (1.0 - dones) * self.cfg.discount_factor_g * next_q

        loss = self.criterion(q_values, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        return float(loss.item())

    def train(self, episodes: int = 500):
        best_reward = -float("inf")
        reward_history = []

        for ep in range(1, episodes + 1):
            state, _ = self.env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # store transition
                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                ep_reward += reward

                _ = self.optimize()

                # target network sync
                if self.steps_done % self.cfg.network_sync_rate == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                # epsilon decay
                if self.epsilon > self.cfg.epsilon_min:
                    self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)

            reward_history.append(ep_reward)
            if ep_reward > best_reward:
                best_reward = ep_reward
                self.save(f"saved_models/best.pt")

            print(f"[Episode {ep:4d}] reward={ep_reward:5.1f}  eps={self.epsilon:.3f}")

            if ep_reward >= self.cfg.stop_on_reward:
                print("Reached target reward. Stopping training.")
                break

        # save last
        self.save(f"saved_models/last.pt")
        return reward_history

    def evaluate(self, episodes: int = 5, model_path: str | None = None, render: bool = True):
        if model_path:
            self.load(model_path)

        env = self.eval_env if render else self.env
        scores = []
        for ep in range(episodes):
            state, _ = env.reset()
            done = False
            ep_reward = 0
            while not done:
                with torch.no_grad():
                    s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                    action = int(self.policy_net(s).argmax(dim=1).item())
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_reward += reward
            scores.append(ep_reward)
            print(f"[Eval {ep+1}/{episodes}] reward={ep_reward}")
        return scores

    def save(self, path: str):
        torch.save({
            "model_state_dict": self.policy_net.state_dict(),
            "cfg": self.cfg.__dict__,
        }, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(data["model_state_dict"])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.eval()
