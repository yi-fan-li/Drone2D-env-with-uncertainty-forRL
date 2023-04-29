from stable_baselines3 import PPO
import gymnasium as gym
import pandas as pd
import numpy as np
import Drone2D_ENV_forRL_with_Uncertainty_env

constant_wind_env = gym.make("Drone2D-Uncertain-v0", render_sim = False, Sensor_noise_level = "none",
                   Actuator_noise_level = "none", Environmental_disturbance = "constant")

constant_wind_model = PPO("MlpPolicy", constant_wind_env)

constant_wind_model.learn(total_timesteps= 250000)

constant_wind_rewards = []

for i in range(100):
    terminated = False
    truncated = False
    obs, info = constant_wind_env.reset()
    episode_reward = []
    while not terminated and not truncated:
        action, _states = constant_wind_model.predict(obs)
        obs, reward, terminated, truncated, info = constant_wind_env.step(action)
        episode_reward.append(reward)
    episode_reward = np.array(episode_reward)
    constant_wind_rewards.append(np.sum(episode_reward))


constant_wind_model.save("C:/Users/their/Desktop/Mcgill University/Session 10/COMP 579/Project/Agents/constant_wind_model")

pd.DataFrame(constant_wind_rewards).to_csv("C:/Users/their/Desktop/Mcgill University/Session 10/COMP 579/Project/Raw data/constant_wind_rewards.csv", header=None, index=None)

random_wind_env = gym.make("Drone2D-Uncertain-v0", render_sim = False, Sensor_noise_level = "none",
                   Actuator_noise_level = "none", Environmental_disturbance = "random")

random_wind_model = PPO("MlpPolicy", random_wind_env)

random_wind_model.learn(total_timesteps= 250000)

random_wind_rewards = []

for i in range(100):
    terminated = False
    truncated = False
    obs, info = random_wind_env.reset()
    episode_reward = []
    while not terminated and not truncated:
        action, _states = random_wind_model.predict(obs)
        obs, reward, terminated, truncated, info = random_wind_env.step(action)
        episode_reward.append(reward)
    episode_reward = np.array(episode_reward)
    random_wind_rewards.append(np.sum(episode_reward))

random_wind_model.save("C:/Users/their/Desktop/Mcgill University/Session 10/COMP 579/Project/Agents/random_wind_model")

pd.DataFrame(random_wind_rewards).to_csv("C:/Users/their/Desktop/Mcgill University/Session 10/COMP 579/Project/Raw data/random_wind_rewards.csv", header=None, index=None)
