from gymnasium.envs.registration import register
from Drone2D_ENV_forRL_with_Uncertainty_env.Drone2dEnv_with_uncertainty import Drone2dEnv_with_uncertainty


register(
    id='Drone2D-Uncertain-v0',
    entry_point='Drone2D_ENV_forRL_with_Uncertainty_env:Drone2dEnv_with_uncertainty',
    kwargs={'render_sim': False, 'render_path': True, 'render_shade': True,
            'shade_distance': 75, 'time_in_second': 10}
)