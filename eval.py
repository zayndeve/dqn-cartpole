# eval.py
import time
import argparse
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# Agent + config live in train.py (since we merged agent.py)
from train import Agent, load_config

# === Same normalization as used during training ===
# Bounds roughly follow CartPole observation limits:
# [cart_position≈±2.4, cart_velocity≈±3.0, pole_angle≈±12deg(≈0.2095rad), pole_tip_velocity≈±3.5]
_BOUNDS = np.array([2.4, 3.0, 0.2095, 3.5], dtype=np.float32)

def _norm(s: np.ndarray) -> np.ndarray:
    """Normalize observation by fixed bounds and clip to [-1, 1]."""
    return np.clip(s / _BOUNDS, -1.0, 1.0)


def run_episode(env, policy_net, render=False, sleep=False, hold=0, margin=0.0):
    """
    Run a single evaluation episode and return the number of steps survived.

    Args:
        env: Gymnasium environment (already created with appropriate render_mode).
        policy_net: Trained PyTorch network that outputs Q-values.
        render (bool): If True, render frames (only meaningful with render_mode="human").
        sleep (bool): If True, throttle the loop to ~60 FPS for human viewing.
        hold (int): Action smoothing—repeat the chosen action for K steps (0 disables).
        margin (float): Hysteresis—switch to a new greedy action only if its Q-value
                        exceeds the previous action's Q by more than this margin.

    Behavior:
        - Uses the same observation normalization as training.
        - Supports Gymnasium's (terminated, truncated) termination API.
        - Optional cosmetic smoothing (hold/margin) can make videos look steadier.
    """
    obs, _ = env.reset()
    obs = _norm(obs)
    done, steps = False, 0
    policy_net.eval()  # disable dropout/batchnorm behavior if present

    last_action = None  # last selected action (for smoothing)
    hold_left = 0       # how many steps remain to keep last_action

    while not done:
        if hold_left > 0 and last_action is not None:
            # Continue holding the previous action
            a = last_action
            hold_left -= 1
        else:
            # Select (possibly smoothed) greedy action from Q-network
            with torch.no_grad():
                s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # shape: (1, state_dim)
                q = policy_net(s)                                        # shape: (1, action_dim)
                a_star = int(q.argmax(1).item())                         # greedy action

                if last_action is None or margin <= 0.0:
                    # No hysteresis: take greedy action
                    a = a_star
                else:
                    # Hysteresis: switch only if advantage over previous is significant
                    q_star = float(q[0, a_star].item())
                    q_prev = float(q[0, last_action].item())
                    a = a_star if (q_star - q_prev) > margin else last_action

            last_action = a
            # If we just (re)selected an action, we will hold it for (hold-1) more steps
            hold_left = max(0, hold - 1)

        # Step environment (Gymnasium returns terminated/truncated separately)
        obs, _, term, trunc, _ = env.step(a)
        obs = _norm(obs)
        done = term or trunc  # episode ends if either condition is True

        if render:
            env.render()
        if sleep:
            time.sleep(1 / 60)  # limit FPS for readability

        steps += 1

    return steps


def main():
    # CLI arguments for flexible evaluation and recording
    ap = argparse.ArgumentParser()
    ap.add_argument("profile", help="YAML profile key, e.g., 'cartpole1'")
    ap.add_argument("--yaml", default="hyperparameters.yml", help="Path to YAML config")
    ap.add_argument("--model", default="saved_models/best.pt", help="Path to model checkpoint")
    ap.add_argument("--human", action="store_true", help="Render interactive window")
    ap.add_argument("--video", action="store_true", help="Record MP4s to videos/ via RecordVideo")
    ap.add_argument("--until-500", action="store_true", help="Repeat episodes until one reaches 500 steps")

    # Optional action smoothing (cosmetic; small values recommended)
    ap.add_argument("--hold", type=int, default=0, help="Repeat chosen action for K steps (0 disables)")
    ap.add_argument("--margin", type=float, default=0.0, help="Hysteresis margin on Q to reduce chattering")

    args = ap.parse_args()

    # Load config and agent; restore trained policy network
    cfg = load_config(args.yaml, args.profile)
    agent = Agent(cfg)
    agent.load(args.model)

    # Create environment with appropriate rendering/recording setup
    if args.video:
        # RecordVideo needs render_mode="rgb_array" to capture frames
        env = RecordVideo(
            gym.make(cfg.env_id, render_mode="rgb_array"),
            video_folder="videos",
            name_prefix="cartpole"
        )
    elif args.human:
        # On-screen rendering (no video saved)
        env = gym.make(cfg.env_id, render_mode="human")
    else:
        # Headless eval (fastest)
        env = gym.make(cfg.env_id)

    target = 500  # CartPole-v1 maximum steps per episode
    while True:
        steps = run_episode(
            env, agent.policy_net,
            render=args.human,
            sleep=args.human,   # throttle only when rendering to a window
            hold=args.hold,
            margin=args.margin
        )
        print(f"steps={steps}")

        # If not looping to 500 or already achieved 500, stop
        if not args.until_500 or steps >= target:
            break

        # Small pause between trials to keep recordings/readability clean
        time.sleep(0.5)

    # Graceful close (some renderers need a brief delay before close)
    if args.human:
        time.sleep(0.5)
    env.close()


if __name__ == "__main__":
    main()
