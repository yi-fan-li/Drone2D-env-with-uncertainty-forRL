
from stable_baselines3 import PPO
import gymnasium as gym

import Drone2D_ENV_forRL_with_Uncertainty_env

env = gym.make("Drone2D-Uncertain-v0", render_sim = False, Sensor_noise_level = "none",
                   Actuator_noise_level = "none", Environmental_disturbance = "none")

model = PPO("MlpPolicy", env)

model.learn(total_timesteps=10000)


vec_env = gym.make("Drone2D-Uncertain-v0", render_sim = True, Sensor_noise_level = "none",
                   Actuator_noise_level = "none", Environmental_disturbance = "none")
obs, info = vec_env.reset()

for i in range(10):
    terminated = False
    truncated = False
    obs, info = vec_env.reset()
    while not terminated and not truncated:
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, info = vec_env.step(action)
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

