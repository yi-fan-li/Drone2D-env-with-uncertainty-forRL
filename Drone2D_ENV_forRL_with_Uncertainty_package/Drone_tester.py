import Drone2D_ENV_forRL_with_Uncertainty_env
import gymnasium as gym

newenv = gym.make("Drone2D-Uncertain-v0", render_sim = True, Sensor_noise_level = "medium",
                   Actuator_noise_level = "none", Environmental_disturbance = "constant")

obs, info = newenv.reset()

from stable_baselines import PPO

model = PPO("MlpPolicy", "CartPole-v1").learn(10_000)