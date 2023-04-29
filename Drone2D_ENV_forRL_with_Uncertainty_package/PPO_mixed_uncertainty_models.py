from stable_baselines3 import PPO
import gymnasium as gym
import pandas as pd
import numpy as np
import Drone2D_ENV_forRL_with_Uncertainty_env

low__sensor_actuator_uncertainty_env = gym.make("Drone2D-Uncertain-v0", render_sim = False, Sensor_noise_level = "low",
                   Actuator_noise_level = "low", Environmental_disturbance = "none")

low_sensor_actuator_uncertainty_model = PPO("MlpPolicy", low__sensor_actuator_uncertainty_env)

low_sensor_actuator_uncertainty_model.learn(total_timesteps= 250000)

low_sensor_actuator_uncertainty_rewards = []

for i in range(100):
    terminated = False
    truncated = False
    obs, info = low__sensor_actuator_uncertainty_env.reset()
    episode_reward = []
    while not terminated and not truncated:
        action, _states = low_sensor_actuator_uncertainty_model.predict(obs)
        obs, reward, terminated, truncated, info = low__sensor_actuator_uncertainty_env.step(action)
        episode_reward.append(reward)
    episode_reward = np.array(episode_reward)
    low_sensor_actuator_uncertainty_rewards.append(np.sum(episode_reward))


low_sensor_actuator_uncertainty_model.save("C:/Users/their/Desktop/Mcgill University/Session 10/COMP 579/Project/Agents/low_sensor_actuator_uncertainty_model")

pd.DataFrame(low_sensor_actuator_uncertainty_rewards).to_csv("C:/Users/their/Desktop/Mcgill University/Session 10/COMP 579/Project/Raw data/low_sensor_actuator_uncertainty_rewards.csv", header=None, index=None)


medium_sensor_actuator_uncertainty_env = gym.make("Drone2D-Uncertain-v0", render_sim = False, Sensor_noise_level = "medium",
                   Actuator_noise_level = "medium", Environmental_disturbance = "none")

medium_sensor_actuator_uncertainty_model = PPO("MlpPolicy", medium_sensor_actuator_uncertainty_env)

medium_sensor_actuator_uncertainty_model.learn(total_timesteps= 250000)

medium_sensor_actuator_uncertainty_rewards = []

for i in range(100):
    terminated = False
    truncated = False
    obs, info = medium_sensor_actuator_uncertainty_env.reset()
    episode_reward = []
    while not terminated and not truncated:
        action, _states = medium_sensor_actuator_uncertainty_model.predict(obs)
        obs, reward, terminated, truncated, info = medium_sensor_actuator_uncertainty_env.step(action)
        episode_reward.append(reward)
    episode_reward = np.array(episode_reward)
    medium_sensor_actuator_uncertainty_rewards.append(np.sum(episode_reward))


medium_sensor_actuator_uncertainty_model.save("C:/Users/their/Desktop/Mcgill University/Session 10/COMP 579/Project/Agents/medium_sensor_actuator_uncertainty_model")

pd.DataFrame(medium_sensor_actuator_uncertainty_rewards).to_csv("C:/Users/their/Desktop/Mcgill University/Session 10/COMP 579/Project/Raw data/medium_sensor_actuator_uncertainty_rewards.csv", header=None, index=None)


high_sensor_actuator_uncertainty_env = gym.make("Drone2D-Uncertain-v0", render_sim = False, Sensor_noise_level = "high",
                   Actuator_noise_level = "high", Environmental_disturbance = "none")
high_sensor_actuator_uncertainty_model = PPO("MlpPolicy", high_sensor_actuator_uncertainty_env)

high_sensor_actuator_uncertainty_model.learn(total_timesteps= 250000)

high_sensor_actuator_uncertainty_rewards = []

for i in range(100):
    terminated = False
    truncated = False
    obs, info = high_sensor_actuator_uncertainty_env.reset()
    episode_reward = []
    while not terminated and not truncated:
        action, _states = high_sensor_actuator_uncertainty_model.predict(obs)
        obs, reward, terminated, truncated, info = high_sensor_actuator_uncertainty_env.step(action)
        episode_reward.append(reward)
    episode_reward = np.array(episode_reward)
    high_sensor_actuator_uncertainty_rewards.append(np.sum(episode_reward))

high_sensor_actuator_uncertainty_model.save("C:/Users/their/Desktop/Mcgill University/Session 10/COMP 579/Project/Agents/high_sensor_actuator_uncertainty_model")

pd.DataFrame(high_sensor_actuator_uncertainty_rewards).to_csv("C:/Users/their/Desktop/Mcgill University/Session 10/COMP 579/Project/Raw data/high_sensor_actuator_uncertainty_rewards.csv", header=None, index=None)
