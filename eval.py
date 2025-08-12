# eval.py
import os
import time
import argparse
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# After merging, Agent/load_config live in train.py
from train import Agent, load_config


def run_episode(env, policy_net, render=False, sleep=False) -> int:
    """Run one greedy episode and return the number of steps."""
    obs, _ = env.reset()
    done, steps = False, 0
    policy_net.eval()

    # Ensure tensor is on the same device as the network
    device = next(policy_net.parameters()).device

    while not done:
        with torch.no_grad():
            s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            a = int(policy_net(s).argmax(1).item())
        obs, _, term, trunc, _ = env.step(a)
        done = term or trunc
        if render:
            env.render()
        if sleep:
            time.sleep(1 / 60)
        steps += 1
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile", help="config profile in hyperparameters.yml, e.g., cartpole1")
    ap.add_argument("--yaml", default="hyperparameters.yml")
    ap.add_argument("--model", default="saved_models/best.pt")
    ap.add_argument("--human", action="store_true", help="render a window")
    ap.add_argument("--video", action="store_true", help="save MP4 to videos/")
    ap.add_argument("--until-500", dest="until_500", action="store_true",
                    help="repeat until an episode hits 500 steps")
    args = ap.parse_args()

    cfg = load_config(args.yaml, args.profile)
    agent = Agent(cfg)
    agent.load(args.model)  # loads weights, sets eval()

    # Choose env based on flags
    if args.video:
        os.makedirs("videos", exist_ok=True)
        env = gym.make(cfg.env_id, render_mode="rgb_array")
        env = RecordVideo(env, video_folder="videos", name_prefix="cartpole")
    elif args.human:
        env = gym.make(cfg.env_id, render_mode="human")
    else:
        env = gym.make(cfg.env_id)  # headless

    target = 500
    while True:
        steps = run_episode(env, agent.policy_net, render=args.human, sleep=args.human)
        print(f"steps={steps}")
        if not args.until_500 or steps >= target:
            break
        time.sleep(0.5)  # small pause before retry

    if args.human:
        time.sleep(0.5)
    env.close()


if __name__ == "__main__":
    main()
