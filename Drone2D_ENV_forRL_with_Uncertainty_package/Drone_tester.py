import Drone2D_ENV_forRL_with_Uncertainty_env
import gymnasium as gym

newenv = gym.make("Drone2D-Uncertain-v0", render_sim = True, Sensor_noise_level = "none",
                   Actuator_noise_level = "none", Environmental_disturbance = "none")

obs, info = newenv.reset()


obs, reward, truncated, terminated, info = newenv.step([1000,1000])
obs, reward, truncated, terminated, info = newenv.step([1000,1000])

print(obs)