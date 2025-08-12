# eval.py
import time
import argparse
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# Agent + config live in train.py (since we merged agent.py)
from train import Agent, load_config

# === Same normalization as training ===
_BOUNDS = np.array([2.4, 3.0, 0.2095, 3.5], dtype=np.float32)
def _norm(s: np.ndarray) -> np.ndarray:
    return np.clip(s / _BOUNDS, -1.0, 1.0)


def run_episode(env, policy_net, render=False, sleep=False, hold=0, margin=0.0):
    """
    Run one episode.
    - hold: keep the chosen action for K steps (0 = off)
    - margin: only switch actions if new Q is better than previous by this margin
    """
    obs, _ = env.reset()
    obs = _norm(obs)
    done, steps = False, 0
    policy_net.eval()

    last_action = None   # previous action (for smoothing)
    hold_left = 0        # remaining steps to hold current action

    while not done:
        if hold_left > 0 and last_action is not None:
            a = last_action
            hold_left -= 1
        else:
            with torch.no_grad():
                s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                q = policy_net(s)  # shape (1, action_dim)
                a_star = int(q.argmax(1).item())

                if last_action is None or margin <= 0.0:
                    a = a_star
                else:
                    q_star = float(q[0, a_star].item())
                    q_prev = float(q[0, last_action].item())
                    # switch only if significantly better
                    a = a_star if (q_star - q_prev) > margin else last_action

            last_action = a
            hold_left = max(0, hold - 1)

        obs, _, term, trunc, _ = env.step(a)
        obs = _norm(obs)
        done = term or trunc

        if render:
            env.render()
        if sleep:
            time.sleep(1/60)

        steps += 1

    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile", help="YAML profile, e.g. cartpole1")
    ap.add_argument("--yaml", default="hyperparameters.yml")
    ap.add_argument("--model", default="saved_models/best.pt")
    ap.add_argument("--human", action="store_true", help="render a window")
    ap.add_argument("--video", action="store_true", help="save MP4 to videos/")
    ap.add_argument("--until-500", action="store_true", help="repeat until an episode hits 500")

    # Smoothing controls (cosmetic; use small values)
    ap.add_argument("--hold", type=int, default=0, help="hold each chosen action for K steps (0=off)")
    ap.add_argument("--margin", type=float, default=0.0, help="only switch actions if Q-advantage exceeds this margin")

    args = ap.parse_args()

    cfg = load_config(args.yaml, args.profile)
    agent = Agent(cfg)
    agent.load(args.model)

    # Choose env based on flags
    if args.video:
        env = RecordVideo(
            gym.make(cfg.env_id, render_mode="rgb_array"),
            video_folder="videos",
            name_prefix="cartpole"
        )
    elif args.human:
        env = gym.make(cfg.env_id, render_mode="human")
    else:
        env = gym.make(cfg.env_id)

    target = 500
    while True:
        steps = run_episode(
            env, agent.policy_net,
            render=args.human,
            sleep=args.human,
            hold=args.hold,
            margin=args.margin
        )
        print(f"steps={steps}")
        if not args.until_500 or steps >= target:
            break
        time.sleep(0.5)

    if args.human:
        time.sleep(0.5)
    env.close()


if __name__ == "__main__":
    main()
