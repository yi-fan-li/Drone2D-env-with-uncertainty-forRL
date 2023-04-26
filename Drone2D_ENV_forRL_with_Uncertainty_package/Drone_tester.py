import Drone2D_ENV_forRL_with_Uncertainty_env
import gymnasium as gym

newenv = gym.make("Drone2D-Uncertain-v0", render_sim = True, Sensor_noise_level = "none",
                   Actuator_noise_level = "none", Environmental_disturbance = "none")

obs, info = newenv.reset()

for i in range (100):
    obs, reward, terminated, truncated, info = newenv.step([800,700])
print(obs)