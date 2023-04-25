import Drone2D_ENV_forRL_with_Uncertainty_env
import gymnasium as gym

print(gym.registry.keys())

newenv = gym.make("Drone2D-Uncertain-v0")

newenv.reset()