# eval.py
import time, argparse, torch, gymnasium as gym
from gymnasium.wrappers import RecordVideo
from agent import Agent, load_config

def run_episode(env, policy_net, render=False, sleep=False):
    obs, _ = env.reset()
    done, steps = False, 0
    policy_net.eval()
    while not done:
        with torch.no_grad():
            a = int(policy_net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).argmax(1).item())
        obs, _, term, trunc, _ = env.step(a)
        done = term or trunc
        if render: env.render()
        if sleep: time.sleep(1/60)
        steps += 1
    return steps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--yaml", default="hyperparameters.yml")
    ap.add_argument("--model", default="saved_models/best.pt")
    ap.add_argument("--human", action="store_true", help="pop up a window")
    ap.add_argument("--video", action="store_true", help="save MP4 to videos/")
    ap.add_argument("--until-500", action="store_true", help="repeat until an episode hits 500")
    args = ap.parse_args()

    cfg = load_config(args.yaml, args.profile)
    agent = Agent(cfg)
    agent.load(args.model)

    # choose env based on flags
    if args.video:
        env = RecordVideo(gym.make(cfg.env_id, render_mode="rgb_array"),
                          video_folder="videos", name_prefix="cartpole")
    elif args.human:
        env = gym.make(cfg.env_id, render_mode="human")
    else:
        env = gym.make(cfg.env_id)  # headless, no window/video

    target = 500
    while True:
        steps = run_episode(env, agent.policy_net, render=args.human, sleep=args.human)
        print(f"steps={steps}")
        if not args.until_500 or steps >= target:
            break
        time.sleep(1)  # small pause before next try

    if args.human: time.sleep(1)
    env.close()

if __name__ == "__main__":
    main()
