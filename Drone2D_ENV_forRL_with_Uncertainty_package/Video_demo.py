import gymnasium as gym
import pandas as pd
import numpy as np
import Drone2D_ENV_forRL_with_Uncertainty_env
import matplotlib.pyplot as plot
from stable_baselines3 import PPO

video_env = gym.make("Drone2D-Uncertain-v0", render_sim = True, Sensor_noise_level = "none",
                   Actuator_noise_level = "none", Environmental_disturbance = "none")

obs, info = video_env.reset()
action = [-0.11,-0.11]
action2 = [0.2,-0.1]

# for i in range (300):
#     video_env.step(action)

# for i in range(100):
#     video_env.step(action2)


# video_env_windy = gym.make("Drone2D-Uncertain-v0", render_sim = True, Sensor_noise_level = "none",
#                    Actuator_noise_level = "none", Environmental_disturbance = "random")

# video_env_windy.reset()

# for i in range (300):
#     video_env_windy.step(action)

# video_model = PPO("MlpPolicy", video_env)

# video_model.learn(total_timesteps= 1000)

# for i in range(10):
#     terminated = False
#     truncated = False
#     obs, info = video_env.reset()
#     while not terminated and not truncated:
#         action, _states = video_model.predict(obs)
#         obs, reward, terminated, truncated, info = video_env.step(action)
                                                                
# no_uncertainty_model = PPO.load("C:/Users/their/Desktop/Mcgill University/Session 10/COMP 579/Project/Agents/no_uncertainty_model")

# for i in range(10):
#     terminated = False
#     truncated = False
#     obs, info = video_env.reset()
#     while not terminated and not truncated:
#         action, _states = no_uncertainty_model.predict(obs)
#         obs, reward, terminated, truncated, info = video_env.step(action)

# low_sensor_uncertainty_model = PPO.load("C:/Users/their/Desktop/Mcgill University/Session 10/COMP 579/Project/Agents/low_sensor_uncertainty_model")

# for i in range(10):
#     terminated = False
#     truncated = False
#     obs, info = video_env.reset()
#     while not terminated and not truncated:
#         action, _states = low_sensor_uncertainty_model.predict(obs)
#         obs, reward, terminated, truncated, info = video_env.step(action)

# high_sensor_uncertainty_model = PPO.load("C:/Users/their/Desktop/Mcgill University/Session 10/COMP 579/Project/Agents/high_sensor_uncertainty_model")

# for i in range(10):
#     terminated = False
#     truncated = False
#     obs, info = video_env.reset()
#     while not terminated and not truncated:
#         action, _states = high_sensor_uncertainty_model.predict(obs)
#         obs, reward, terminated, truncated, info = video_env.step(action)



# low_actuator_uncertainty_model = PPO.load("C:/Users/their/Desktop/Mcgill University/Session 10/COMP 579/Project/Agents/low_actuator_uncertainty_model")

# for i in range(10):
#     terminated = False
#     truncated = False
#     obs, info = video_env.reset()
#     while not terminated and not truncated:
#         action, _states = low_actuator_uncertainty_model.predict(obs)
#         obs, reward, terminated, truncated, info = video_env.step(action)


# high_actuator_uncertainty_model = PPO.load("C:/Users/their/Desktop/Mcgill University/Session 10/COMP 579/Project/Agents/high_actuator_uncertainty_model")

# for i in range(10):
#     terminated = False
#     truncated = False
#     obs, info = video_env.reset()
#     while not terminated and not truncated:
#         action, _states = high_actuator_uncertainty_model.predict(obs)
#         obs, reward, terminated, truncated, info = video_env.step(action)


# high_sensor_actuator_uncertainty_model = PPO.load("C:/Users/their/Desktop/Mcgill University/Session 10/COMP 579/Project/Agents/high_sensor_actuator_uncertainty_model")

# for i in range(10):
#     terminated = False
#     truncated = False
#     obs, info = video_env.reset()
#     while not terminated and not truncated:
#         action, _states = high_sensor_actuator_uncertainty_model.predict(obs)
#         obs, reward, terminated, truncated, info = video_env.step(action)




# constant_wind_model = PPO.load("C:/Users/their/Desktop/Mcgill University/Session 10/COMP 579/Project/Agents/constant_wind_model")

# for i in range(10):
#     terminated = False
#     truncated = False
#     obs, info = video_env.reset()
#     while not terminated and not truncated:
#         action, _states = constant_wind_model.predict(obs)
#         obs, reward, terminated, truncated, info = video_env.step(action)


random_wind_model = PPO.load("C:/Users/their/Desktop/Mcgill University/Session 10/COMP 579/Project/Agents/random_wind_model")

for i in range(10):
    terminated = False
    truncated = False
    obs, info = video_env.reset()
    while not terminated and not truncated:
        action, _states = random_wind_model.predict(obs)
        obs, reward, terminated, truncated, info = video_env.step(action)