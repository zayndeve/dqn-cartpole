import argparse
from agent import Agent, load_config
from utils import save_plot  # <- add this import
import os

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
    rewards = agent.train(episodes=args.episodes)  # returns reward history

    # Save learning curve PNG
    png_path = os.path.join(args.outdir, "learning_curve.png")
    save_plot(rewards, png_path)
    print(f"[OK] Saved learning curve → {png_path}")
    print(f"[INFO] Weights are in {args.outdir}/best.pt and {args.outdir}/last.pt")

if __name__ == "__main__":
    main()
