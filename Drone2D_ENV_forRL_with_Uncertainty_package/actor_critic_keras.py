
from stable_baselines3 import PPO
import gymnasium as gym
import pandas as pd
import numpy as np
import Drone2D_ENV_forRL_with_Uncertainty_env

no_uncertainty_env = gym.make("Drone2D-Uncertain-v0", render_sim = False, Sensor_noise_level = "none",
                   Actuator_noise_level = "none", Environmental_disturbance = "none")

no_uncertainty_model = PPO("MlpPolicy", no_uncertainty_env)

no_uncertainty_model.learn(total_timesteps= 250000)

no_uncertainty_rewards = []

for i in range(100):
    terminated = False
    truncated = False
    obs, info = no_uncertainty_env.reset()
    episode_reward = []
    while not terminated and not truncated:
        action, _states = no_uncertainty_model.predict(obs)
        obs, reward, terminated, truncated, info = no_uncertainty_env.step(action)
        episode_reward.append(reward)
    episode_reward = np.array(episode_reward)
    no_uncertainty_rewards.append(np.sum(episode_reward))


no_uncertainty_model.save("C:/Users/their/Desktop/Mcgill University/Session 10/COMP 579/Project/Agents/no_uncertainty_model")

pd.DataFrame(no_uncertainty_rewards).to_csv("C:/Users/their/Desktop/Mcgill University/Session 10/COMP 579/Project/Raw data/no_uncertainty_rewards.csv", header=None, index=None)

# low_sensor_uncertainty_env = gym.make("Drone2D-Uncertain-v0", render_sim = False, Sensor_noise_level = "low",
#                    Actuator_noise_level = "none", Environmental_disturbance = "none")

# low_sensor_uncertainty_model = PPO("MlpPolicy", low_sensor_uncertainty_env)

# low_sensor_uncertainty_model.learn(total_timesteps= 250000)

# low_sensor_uncertainty_rewards = []

# for i in range(100):
#     terminated = False
#     truncated = False
#     obs, info = low_sensor_uncertainty_env.reset()
#     episode_reward = []
#     while not terminated and not truncated:
#         action, _states = low_sensor_uncertainty_model.predict(obs)
#         obs, reward, terminated, truncated, info = low_sensor_uncertainty_env.step(action)
#         episode_reward.append(reward)
#     episode_reward = np.array(episode_reward)
#     low_sensor_uncertainty_rewards.append(np.sum(episode_reward))

# low_sensor_uncertainty_model.save("C:/Users/their/Desktop/Mcgill University/Session 10/COMP 579/Project/Agents/low_sensor_uncertainty_model")

# pd.DataFrame(low_sensor_uncertainty_rewards).to_csv("C:/Users/their/Desktop/Mcgill University/Session 10/COMP 579/Project/Raw data/low_sensor_uncertainty_rewards.csv", header=None, index=None)


vec_env = gym.make("Drone2D-Uncertain-v0", render_sim = True, Sensor_noise_level = "none",
                   Actuator_noise_level = "none", Environmental_disturbance = "none")
obs, info = vec_env.reset()

for i in range(10):
    terminated = False
    truncated = False
    obs, info = vec_env.reset()
    while not terminated and not truncated:
        action, _states = no_uncertainty_model.predict(obs)
        obs, rewards, terminated, truncated, info = vec_env.step(action)

# vec_env = gym.make("Drone2D-Uncertain-v0", render_sim = True, Sensor_noise_level = "low",
#                    Actuator_noise_level = "none", Environmental_disturbance = "none")
# obs, info = vec_env.reset()

# for i in range(10):
#     terminated = False
#     truncated = False
#     obs, info = vec_env.reset()
#     while not terminated and not truncated:
#         action, _states = low_sensor_uncertainty_model.predict(obs)
#         obs, rewards, terminated, truncated, info = vec_env.step(action)
"""
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save("a2c_cartpole")

del model # remove to demonstrate saving and loading

model = A2C.load("a2c_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
"""

