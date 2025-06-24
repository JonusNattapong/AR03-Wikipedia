!pip install stable-baselines3 gymnasium

import gymnasium as gym
from stable_baselines3 import PPO

# สร้าง environment
env = gym.make('CartPole-v1')

# สร้าง PPO โมเดล
model = PPO('MlpPolicy', env, verbose=1)

# ฝึกโมเดล
model.learn(total_timesteps=10_000)

# ทดสอบโมเดล
obs, info = env.reset()
for _ in range(500):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done or truncated:
        obs, info = env.reset()
env.close()
