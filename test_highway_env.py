import gymnasium as gym
import highway_env

env = gym.make("highway-v0")
obs, info = env.reset()

print("✅ Environment reset successful")
print("Observation shape:", obs.shape)
print("Action space:", env.action_space)

done = False
steps = 0
while not done and steps < 100:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    steps += 1

env.close()
print("✅ Random rollout completed")
