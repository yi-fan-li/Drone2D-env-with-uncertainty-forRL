import pymunk
import pymunk.pygame_util
from pymunk import Vec2d
import numpy as np
import pygame

import Drone2D_ENV_forRL_with_Uncertainty_env.Drone_physic as Drone
import Drone2D_ENV_forRL_with_Uncertainty_env.Event_handler as Event_handler

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import random
import os

class Drone2dEnv_with_uncertainty(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}
    """
    render_sim: (bool) if true, a graphic is generated
    render_path: (bool) if true, the drone's path is drawn
    render_shade: (bool) if true, the drone's shade is drawn
    shade_distance: (int) distance between consecutive drone's shades
    time_in_second: (int) number of second before truncation
    Sensor_noise_level: (string) the level of noise in the sensor in observation (none, low, medium, or high)
    Actuator_noise_level: (string) the level of noise in the motor thrust (none, low, medium, high)
    Environmental_disturbance: (string) the presence of wind in the environment (none, constant, random)
    """

    def __init__(
            self, 
            render_sim=False, 
            render_path = False, 
            render_shade = False,
            shade_distance = 10,
            time_in_second = 20,
            Sensor_noise_level = "none",
            Actuator_noise_level = "none",
            Environmental_disturbance = "none"
            ):

        #initiate rng
        self.rng = np.random.default_rng(50)


        self.render_sim = render_sim
        self.render_path = render_path
        self.render_shade = render_shade

        if self.render_sim is True:
            self.init_pygame()
            self.flight_path = []
            self.drop_path = []
            self.path_drone_shade = []

        self.init_Drone()
        
        # Parameters
        self.max_time_steps = time_in_second * 60
        self.drone_shade_distance = shade_distance
        self.force_scale = 1200


        #Initial values
        self.first_step = True
        self.terminated = False
        self.truncated = False
        self.info = {}
        self.current_time_step = 0

        
        #Generating target position
        self.landing_target = [400,600]


        if self.render_sim == True:
            self.render()

        Sensor_noise_level.lower()
        Sensor_noise_level.strip()
        #Checking input for noise level for sensor
        valid_noise_level = (Sensor_noise_level == "none" or Sensor_noise_level == "low" or 
                            Sensor_noise_level == "medium" or Sensor_noise_level == "high")

        if valid_noise_level:
            self.Sensor_noise_level = Sensor_noise_level

        Actuator_noise_level.lower()
        Actuator_noise_level.strip()
        #Checking input for noise level for actuator
        valid_actuator_level = (Actuator_noise_level == "none" or Actuator_noise_level == "low" or 
                            Actuator_noise_level == "medium" or Actuator_noise_level == "high")

        if valid_actuator_level:
            self.Actuator_noise_level = Actuator_noise_level

        Environmental_disturbance.lower()
        Environmental_disturbance.strip()
        #Checking the input for the environmental disturbance
        valid_disturbance_level = (Environmental_disturbance == "none" or Environmental_disturbance == "constant" or 
                            Environmental_disturbance == "random")

        if valid_disturbance_level:
            self.Environmental_disturbance = Environmental_disturbance
        
        constant_environmental_disturbance =  self.Environmental_disturbance == "constant"
        if constant_environmental_disturbance:
            self.wind_force = self.rng.uniform(-300,300)

        #Defining spaces for action and observation
        min_action = np.array([-1, -1], dtype=np.float32)
        max_action = np.array([1, 1], dtype=np.float32)
        self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32)
        
        # observation space (velocity_x, velocity_y, angular velocity, angle, x, y )
        min_observation = np.array([-np.inf, -np.inf, -np.inf, -np.pi, -np.inf, -np.inf], dtype=np.float64)
        max_observation = np.array([np.inf, np.inf, np.inf, np.pi, np.inf, np.inf], dtype=np.float64)
        self.observation_space = spaces.Box(low=min_observation, high=max_observation, dtype=np.float64)

    def init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1000, 800))
        pygame.display.set_caption("Drone2d Environment")
        self.clock = pygame.time.Clock()

        script_dir = os.path.dirname(__file__)
        icon_path = os.path.join("img", "icon.png")
        icon_path = os.path.join(script_dir, icon_path)
        pygame.display.set_icon(pygame.image.load(icon_path))

        img_path = os.path.join("img", "shade.png")
        img_path = os.path.join(script_dir, img_path)
        self.shade_image = pygame.image.load(img_path)

    def init_Drone(self):
        self.space = pymunk.Space()
        self.space.gravity = Vec2d(0, -980)

        if self.render_sim is True:
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            self.draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
            pymunk.pygame_util.positive_y_is_up = True

        #Generating drone's starting position
        random_x = random.uniform(200, 800)
        random_y = random.uniform(600, 800)
        angle_rand = random.uniform(-np.pi/4, np.pi/4)
        # angle_rand = 0
        self.drone = Drone.Drone_physic(random_x, random_y, angle_rand, 16, 80, 0.8, 0.4, self.space)

        self.drone_radius = self.drone.drone_radius

        # apply intial force
        self.left_force = self.rng.uniform(600,800)
        self.right_force = self.rng.uniform(600,800)

        self.drone.frame_shape.body.apply_force_at_local_point(Vec2d(- np.sin(angle_rand)*self.left_force, np.cos(angle_rand)*self.left_force), (-self.drone_radius, 0))
        self.drone.frame_shape.body.apply_force_at_local_point(Vec2d(- np.sin(angle_rand)*self.right_force, np.cos(angle_rand)*self.right_force), (self.drone_radius, 0))


    def step(self, action):

        self.left_force = (action[0]/2+0.6)*self.force_scale
        self.right_force = (action[1]/1+0.6)*self.force_scale

        # noise simulation and seeding
        no_actuator_noise = self.Actuator_noise_level == "none"
        low_actuator_noise = self.Actuator_noise_level == "low"
        medium_actuator_noise = self.Actuator_noise_level == "medium"
        high_actuator_noise = self.Actuator_noise_level == "high"

        # low sensor noise is set to around 0.5% of full scale
        if no_actuator_noise:
            left_force = self.left_force
            right_force = self.right_force

        if low_actuator_noise:
            left_force = self.rng.normal(self.left_force, self.left_force * 0.005)
            right_force = self.rng.normal(self.right_force, self.right_force * 0.005)
        
        # medium sensor noise is set to around 1% of full scale
        if medium_actuator_noise:
            left_force = self.rng.normal(self.left_force, self.left_force * 0.01)
            right_force = self.rng.normal(self.right_force, self.right_force * 0.01)

        # high sensor noise is set to around 5% of full scale
        if high_actuator_noise:
            left_force = self.rng.normal(self.left_force, self.left_force * 0.05)
            right_force = self.rng.normal(self.right_force, self.right_force * 0.05)

        left_force = np.absolute(left_force)
        right_force = np.absolute(right_force)

        # get observations
        obs = self.get_observation()
        velocity_x, velocity_y, angular_velocity, angle, x, y = obs

        # apply the force of the observation
        self.drone.frame_shape.body.apply_force_at_local_point(Vec2d(- np.sin(angle)*left_force, np.cos(angle)*left_force), (-self.drone_radius, 0))
        self.drone.frame_shape.body.apply_force_at_local_point(Vec2d(- np.sin(angle)*right_force, np.cos(angle)*right_force), (self.drone_radius, 0))

        #apply environmental disturbance
        constant_environmental_disturbance =  self.Environmental_disturbance == "constant"
        random_environmental_disturbance = self.Environmental_disturbance == "random"
        
        if constant_environmental_disturbance:
            self.drone.frame_shape.body.apply_force_at_local_point(Vec2d(0, self.wind_force), (self.drone_radius, 0))
        
        if random_environmental_disturbance:
            random_wind_force = self.rng.uniform(-300,300)
            self.drone.frame_shape.body.apply_force_at_local_point(Vec2d(0, random_wind_force), (self.drone_radius, 0))

        self.space.step(1.0/60)
        self.current_time_step += 1
        self.render()

        #Saving drone's position for drawing
        if self.first_step is True:
            if self.render_sim is True and self.render_path is True: self.add_postion_to_drop_path()
            if self.render_sim is True and self.render_shade is True: self.add_drone_shade()
            if self.render_sim is True and self.render_path is True: self.add_postion_to_flight_path()
            self.first_step = False

        else:
            if self.render_sim is True and self.render_path is True: self.add_postion_to_flight_path()

        if self.render_sim is True and self.render_shade is True:
            x, y = self.drone.frame_shape.body.position
            if np.abs(self.shade_x-x) > self.drone_shade_distance or np.abs(self.shade_y-y) > self.drone_shade_distance:
                self.add_drone_shade()

        #Calulating reward function

        x_dist = (500 - x)**2
        y_dist = y**2
        euclid_dist = np.sqrt(x_dist + y_dist)
        reward_dist = 10/(euclid_dist+1)

        reward_angle = -5 * abs(angle)
        # reward_speed = 50/abs(velocity_x + 1) + 50/abs(velocity_y + 1)
        reward_speed = -(abs(velocity_x) + abs(velocity_y))/200

        # reward_thrust = -(left_force + right_force)/60000


        #Stops episode, when drone is out of range or overlaps
        out_of_control = np.abs(angle) > np.pi/2
        out_of_bound = x < 0 or x > 1000 or y > 800
        crashing = y < 8

        # Rewards calculation
        #Check the reward if they landed
        in_landing_zone = self.landing_target[0] < x < self.landing_target[1] and 0 < y < 16
        reasonable_landing_speed = abs(velocity_x) < 10 and abs(velocity_y) < 10 and abs(angular_velocity) < 0.2

        # reward = reward_dist + reward_angle + reward_speed + reward_thrust
        reward = reward_dist + reward_angle + reward_speed

        if out_of_control or out_of_bound:
            self.terminated = True
            reward -= 100

        elif in_landing_zone and reasonable_landing_speed:
            reward += 100
            self.terminated = True

        if not reasonable_landing_speed and crashing:
            reward -= 100
            self.terminated = True

            
        #Stops episode, when time is up
        if self.current_time_step == self.max_time_steps:
            self.truncated = True

        return obs, reward, self.terminated, self.truncated, self.info

    def get_observation(self):
        velocity_x, velocity_y = self.drone.frame_shape.body.velocity_at_local_point((0, 0))
        # velocity_x = np.clip(velocity_x/1330, -1, 1)
        # velocity_y = np.clip(velocity_y/1330, -1, 1)

        omega = self.drone.frame_shape.body.angular_velocity
        # omega = np.clip(omega/11.7, -1, 1)

        alpha = self.drone.frame_shape.body.angle
        # alpha = np.clip(alpha/(np.pi/2), -1, 1)

        x, y = self.drone.frame_shape.body.position
        
        # if x < self.x_target:
        #     distance_x = np.clip((x/self.x_target) - 1, -1, 0)

        # else:
        #     distance_x = np.clip((-x/(self.x_target-800) + self.x_target/(self.x_target-800)) , 0, 1)

        # if y < self.y_target:
        #     distance_y = np.clip((y/self.y_target) - 1, -1, 0)

        # else:
        #     distance_y = np.clip((-y/(self.y_target-800) + self.y_target/(self.y_target-800)) , 0, 1)

        # pos_x = np.clip(x/400.0 - 1, -1, 1)
        # pos_y = np.clip(y/400.0 - 1, -1, 1)

        # return np.array([velocity_x, velocity_y, omega, alpha, distance_x, distance_y, pos_x, pos_y])

        # Noise simulation and rng seeding
        no_sensor_noise = self.Sensor_noise_level == "none"
        low_sensor_noise = self.Sensor_noise_level == "low"
        medium_sensor_noise = self.Sensor_noise_level == "medium"
        high_sensor_noise = self.Sensor_noise_level == "high"

        # low sensor noise is set to around 0.5% of full scale
        if low_sensor_noise:
            x = self.rng.normal(x, 4.0)
            y = self.rng.normal(y, 4.0)
            velocity_x = self.rng.normal(velocity_x, 6.0)
            velocity_y = self.rng.normal(velocity_y, 6.0)
            omega = self.rng.normal(omega, 0.06)
            alpha = self.rng.normal(alpha, 0.008)

        # medium sensor noise is set to around 1% of full scale    
        if medium_sensor_noise:
            x = self.rng.normal(x, 8.0)
            y = self.rng.normal(y, 8.0)
            velocity_x = self.rng.normal(velocity_x, 13.3)
            velocity_y = self.rng.normal(velocity_y, 13.3)
            omega = self.rng.normal(omega, 0.11)
            alpha = self.rng.normal(alpha, 0.0157)
        
        # high sensor noise is set to around 5% of full scale
        if high_sensor_noise:
            x = self.rng.normal(x, 40.0)
            y = self.rng.normal(y, 40.0)
            velocity_x = self.rng.normal(velocity_x, 66.5)
            velocity_y = self.rng.normal(velocity_y, 66.5)
            omega = self.rng.normal(omega, 0.6)
            alpha = self.rng.normal(alpha, 0.08)

        return np.array([velocity_x, velocity_y, omega, alpha, x, y])

    def render(self, mode='human', close=False):
        if self.render_sim is False: return

        Event_handler.pygame_events(self.space, self)
        self.screen.fill((243, 243, 243))
        pygame.draw.rect(self.screen, (24, 114, 139), pygame.Rect(0, 0, 1000, 800), 8)
        pygame.draw.rect(self.screen, (33, 158, 188), pygame.Rect(200, 0, 600, 200), 4)
        pygame.draw.rect(self.screen, (142, 202, 230), pygame.Rect(400, 784, 200, 16), 4)

        #Drawing drone's shade
        if len(self.path_drone_shade):
            for shade in self.path_drone_shade:
                image_rect_rotated = pygame.transform.rotate(self.shade_image, shade[2]*180.0/np.pi)
                shade_image_rect = image_rect_rotated.get_rect(center=(shade[0], 800-shade[1]))
                self.screen.blit(image_rect_rotated, shade_image_rect)

        self.space.debug_draw(self.draw_options)

        #Drawing vectors of motor forces
        vector_scale = 0.05
        l_x_1, l_y_1 = self.drone.frame_shape.body.local_to_world((-self.drone_radius, 0))
        l_x_2, l_y_2 = self.drone.frame_shape.body.local_to_world((-self.drone_radius, 1500*vector_scale))
        pygame.draw.line(self.screen, (179,179,179), (l_x_1, 800-l_y_1), (l_x_2, 800-l_y_2), 4)

        l_x_2, l_y_2 = self.drone.frame_shape.body.local_to_world((-self.drone_radius, self.left_force*vector_scale))
        pygame.draw.line(self.screen, (255,0,0), (l_x_1, 800-l_y_1), (l_x_2, 800-l_y_2), 4)

        r_x_1, r_y_1 = self.drone.frame_shape.body.local_to_world((self.drone_radius, 0))
        r_x_2, r_y_2 = self.drone.frame_shape.body.local_to_world((self.drone_radius, 1500*vector_scale))
        pygame.draw.line(self.screen, (179,179,179), (r_x_1, 800-r_y_1), (r_x_2, 800-r_y_2), 4)

        r_x_2, r_y_2 = self.drone.frame_shape.body.local_to_world((self.drone_radius, self.right_force*vector_scale))
        pygame.draw.line(self.screen, (255,0,0), (r_x_1, 800-r_y_1), (r_x_2, 800-r_y_2), 4)

        # pygame.draw.circle(self.screen, (255, 0, 0), (self.x_target, 800-self.y_target), 5)

        #Drawing drone's path
        if len(self.flight_path) > 2:
            pygame.draw.aalines(self.screen, (16, 19, 97), False, self.flight_path)

        if len(self.drop_path) > 2:
            pygame.draw.aalines(self.screen, (255, 0, 0), False, self.drop_path)

        pygame.display.flip()
        self.clock.tick(60)

    def reset(self):
        self.__init__(self.render_sim, self.render_path, self.render_shade, self.drone_shade_distance,
                      self.max_time_steps/60, self.Sensor_noise_level, 
                      self.Actuator_noise_level, self.Environmental_disturbance)
        return self.get_observation(), self.info

    def close(self):
        pygame.quit()

    # def initial_movement(self):
    #     

    def add_postion_to_drop_path(self):
        x, y = self.drone.frame_shape.body.position
        self.drop_path.append((x, 800-y))

    def add_postion_to_flight_path(self):
        x, y = self.drone.frame_shape.body.position
        self.flight_path.append((x, 800-y))

    def add_drone_shade(self):
        x, y = self.drone.frame_shape.body.position
        self.path_drone_shade.append([x, y, self.drone.frame_shape.body.angle])
        self.shade_x = x
        self.shade_y = y

    # def change_target_point(self, x, y):
    #     self.x_target = x
    #     self.y_target = y


# env = Drone2dEnv_with_uncertainty(render_sim=True)