import gymnasium as gym

env = gym.make('Acrobot-v1')
env = gym.make("LunarLander-v2", render_mode="human")

# reset to get first observation
observation, info = env.reset()

for _ in range(1000):
    # sample random actions rather than agent policy
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()