import Drone2D_ENV_forRL_with_Uncertainty_env
import gymnasium as gym

newenv = gym.make("Drone2D-Uncertain-v0", render_sim = True, Sensor_noise_level = "medium",
                   Actuator_noise_level = "none", Environmental_disturbance = "constant")

obs, info = newenv.reset()

for i in range (10):
    obs, reward, terminated, truncated, info = newenv.step([0,0])
print(obs)