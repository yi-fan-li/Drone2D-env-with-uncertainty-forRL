from stable_baselines3 import PPO
import gymnasium as gym
import pandas as pd
import numpy as np
import Drone2D_ENV_forRL_with_Uncertainty_env

low_actuator_uncertainty_env = gym.make("Drone2D-Uncertain-v0", render_sim = False, Sensor_noise_level = "none",
                   Actuator_noise_level = "low", Environmental_disturbance = "none")

low_actuator_uncertainty_model = PPO("MlpPolicy", low_actuator_uncertainty_env)

low_actuator_uncertainty_model.learn(total_timesteps= 250000)

low_actuator_uncertainty_rewards = []

for i in range(100):
    terminated = False
    truncated = False
    obs, info = low_actuator_uncertainty_env.reset()
    episode_reward = []
    while not terminated and not truncated:
        action, _states = low_actuator_uncertainty_model.predict(obs)
        obs, reward, terminated, truncated, info = low_actuator_uncertainty_env.step(action)
        episode_reward.append(reward)
    episode_reward = np.array(episode_reward)
    low_actuator_uncertainty_rewards.append(np.sum(episode_reward))


low_actuator_uncertainty_model.save("C:/Users/their/Desktop/Mcgill University/Session 10/COMP 579/Project/Agents/low_actuator_uncertainty_model")

pd.DataFrame(low_actuator_uncertainty_rewards).to_csv("C:/Users/their/Desktop/Mcgill University/Session 10/COMP 579/Project/Raw data/low_actuator_uncertainty_rewards.csv", header=None, index=None)


medium_actuator_uncertainty_env = gym.make("Drone2D-Uncertain-v0", render_sim = False, Sensor_noise_level = "none",
                   Actuator_noise_level = "medium", Environmental_disturbance = "none")

medium_actuator_uncertainty_model = PPO("MlpPolicy", medium_actuator_uncertainty_env)

medium_actuator_uncertainty_model.learn(total_timesteps= 250000)

medium_actuator_uncertainty_rewards = []

for i in range(100):
    terminated = False
    truncated = False
    obs, info = medium_actuator_uncertainty_env.reset()
    episode_reward = []
    while not terminated and not truncated:
        action, _states = medium_actuator_uncertainty_model.predict(obs)
        obs, reward, terminated, truncated, info = medium_actuator_uncertainty_env.step(action)
        episode_reward.append(reward)
    episode_reward = np.array(episode_reward)
    medium_actuator_uncertainty_rewards.append(np.sum(episode_reward))


medium_actuator_uncertainty_model.save("C:/Users/their/Desktop/Mcgill University/Session 10/COMP 579/Project/Agents/medium_actuator_uncertainty_model")

pd.DataFrame(medium_actuator_uncertainty_rewards).to_csv("C:/Users/their/Desktop/Mcgill University/Session 10/COMP 579/Project/Raw data/medium_actuator_uncertainty_rewards.csv", header=None, index=None)


high_actuator_uncertainty_env = gym.make("Drone2D-Uncertain-v0", render_sim = False, Sensor_noise_level = "none",
                   Actuator_noise_level = "high", Environmental_disturbance = "none")
high_actuator_uncertainty_model = PPO("MlpPolicy", high_actuator_uncertainty_env)

high_actuator_uncertainty_model.learn(total_timesteps= 250000)

high_actuator_uncertainty_rewards = []

for i in range(100):
    terminated = False
    truncated = False
    obs, info = high_actuator_uncertainty_env.reset()
    episode_reward = []
    while not terminated and not truncated:
        action, _states = high_actuator_uncertainty_model.predict(obs)
        obs, reward, terminated, truncated, info = high_actuator_uncertainty_env.step(action)
        episode_reward.append(reward)
    episode_reward = np.array(episode_reward)
    high_actuator_uncertainty_rewards.append(np.sum(episode_reward))

high_actuator_uncertainty_model.save("C:/Users/their/Desktop/Mcgill University/Session 10/COMP 579/Project/Agents/high_actuator_uncertainty_model")

pd.DataFrame(high_actuator_uncertainty_rewards).to_csv("C:/Users/their/Desktop/Mcgill University/Session 10/COMP 579/Project/Raw data/high_actuator_uncertainty_rewards.csv", header=None, index=None)
