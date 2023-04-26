from stable_baselines3 import PPO
import gymnasium as gym

import Drone2D_ENV_forRL_with_Uncertainty_env

env = gym.make("Drone2D-Uncertain-v0", render_sim = True, Sensor_noise_level = "medium",
                   Actuator_noise_level = "none", Environmental_disturbance = "constant")


model = PPO("MlpPolicy", env)

model.learn(total_timesteps=1800000)
