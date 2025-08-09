import gymnasium as gym

# Create CartPole environment
env = gym.make("CartPole-v1", render_mode="human")

state, info = env.reset()
done = False
total_reward = 0

while not done:
    action = env.action_space.sample()  # choose a random action
    state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print("Episode reward:", total_reward)
env.close()
