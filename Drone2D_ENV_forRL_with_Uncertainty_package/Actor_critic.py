# actor-critic agent (for reference)

import numpy as np
import random
import matplotlib.pyplot as plt
import Drone2D_ENV_forRL_with_Uncertainty_env
import gymnasium as gym
import pygame

class DroneActorCriticAgent:
    def __init__(
            self,
            learning_rate_stateValue: float,
            learning_rate_policy: float,
            discount_factor: float,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (Q), a learning rate

        Args:
            learning_rate: The learning rate
            discount_factor: The discount factor for computing the Q-value
        """
        # np.random.seed (40)

        self.policyWeightLeftupThrust = np.random.default_rng().uniform(-0.001, 0.001, (8, 10))
        self.policyWeightLeftmaintainThrust = np.random.default_rng().uniform(-0.001, 0.001, (8, 10))
        self.policyWeightLeftdownThrust = np.random.default_rng().uniform(-0.001, 0.001, (8, 10))
        self.policyWeightRightupThrust = np.random.default_rng().uniform(-0.001, 0.001, (8, 10))
        self.policyWeightRightmaintainThrust = np.random.default_rng().uniform(-0.001, 0.001, (8, 10))
        self.policyWeightRightdownThrust = np.random.default_rng().uniform(-0.001, 0.001, (8, 10))

        self.stateValueWeight = np.random.default_rng().uniform(-0.001, 0.001, (8, 10))
        self.policyWeight0 = np.random.default_rng().uniform(-0.001, 0.001)
        self.stateValueWeight0 = np.random.default_rng().uniform(-0.001, 0.001)

        self.learning_rate_stateValue = learning_rate_stateValue
        self.learning_rate_policy = learning_rate_policy
        self.discount_factor = discount_factor
        self.prob_pi = np.ones(2)

    def get_featureVector(self, obs):
        velocity_x, velocity_y, angularV, angle, x, y, left_thrust, right_thrust = obs
        featureVector = np.full((8, 10), False, dtype=bool)
        pi = 3.1415926

        dronePosition_xBin = np.linspace(0, 1000, 10)
        dronePosition_yBin = np.linspace(0, 800, 10)
        droneVelocity_xBin = np.linspace(-10, 10, 10)
        droneVelocity_yBin = np.linspace(-10, 10, 10)

        droneAngleBin = np.linspace(-pi, pi, 10)
        droneAngularVBin = np.linspace(-pi, pi, 10)
        DroneLeftThrust = np.linspace(200, 1500, 10)
        DroneRightThrust = np.linspace(200, 1500, 10)

        


        for index, bin in enumerate(dronePosition_xBin):

            inFirstBin = x <= bin and index == 0
            inBetweenBin = x <= bin and x > dronePosition_xBin[index - 1]
            inLastBin = x > dronePosition_xBin[index - 1] and index == 9
            if (inFirstBin or inBetweenBin or inLastBin):
                featureVector[0][index] = True

        for index, bin in enumerate(dronePosition_yBin):

            inFirstBin = y <= bin and index == 0
            inBetweenBin = y <= bin and y > dronePosition_yBin[index - 1]
            inLastBin = y > dronePosition_yBin[index - 1] and index == 9
            if (inFirstBin or inBetweenBin or inLastBin):
                featureVector[1][index] = True

        for index, bin in enumerate(droneVelocity_xBin):

            inFirstBin = velocity_x <= bin and index == 0
            inBetweenBin = velocity_x <= bin and velocity_x > droneVelocity_xBin[index - 1]
            inLastBin = velocity_x > droneVelocity_xBin[index - 1] and index == 9
            if (inFirstBin or inBetweenBin or inLastBin):
                featureVector[2][index] = True

        for index, bin in enumerate(droneVelocity_yBin):

            inFirstBin = velocity_y <= bin and index == 0
            inBetweenBin = velocity_y <= bin and velocity_y > droneVelocity_yBin[index - 1]
            inLastBin = velocity_y > droneVelocity_yBin[index - 1] and index == 9
            if (inFirstBin or inBetweenBin or inLastBin):
                featureVector[3][index] = True

        for index, bin in enumerate(droneAngularVBin):

            inFirstBin = angularV <= bin and index == 0
            inBetweenBin = angularV <= bin and angularV > droneAngularVBin[index - 1]
            inLastBin = angularV > droneAngularVBin[index - 1] and index == 9
            if (inFirstBin or inBetweenBin or inLastBin):
                featureVector[4][index] = True

        for index, bin in enumerate(droneAngleBin):

            inFirstBin = angle <= bin and index == 0
            inBetweenBin = angle <= bin and angle > droneAngleBin[index - 1]
            inLastBin = angle > droneAngleBin[index - 1] and index == 9
            if (inFirstBin or inBetweenBin or inLastBin):
                featureVector[5][index] = True

        for index, bin in enumerate(DroneLeftThrust):

            inFirstBin = left_thrust <= bin and index == 0
            inBetweenBin = left_thrust <= bin and left_thrust > DroneLeftThrust[index - 1]
            inLastBin = left_thrust > DroneLeftThrust[index - 1] and index == 9
            if (inFirstBin or inBetweenBin or inLastBin):
                featureVector[6][index] = True

        for index, bin in enumerate(DroneRightThrust):

            inFirstBin = right_thrust <= bin and index == 0
            inBetweenBin = right_thrust <= bin and right_thrust > DroneRightThrust[index - 1]
            inLastBin = right_thrust > DroneRightThrust[index - 1] and index == 9
            if (inFirstBin or inBetweenBin or inLastBin):
                featureVector[7][index] = True

        return featureVector

    def get_action(self, obs) -> int:
        # Boltzmann policy
        featureVector = self.get_featureVector(obs)
        h_left = self.policyWeight0 + np.dot(
            self.policyWeightLeft.reshape(80), featureVector.reshape(80))
        h_right = self.policyWeight0 + np.dot(
            self.policyWeightRight.reshape(80), featureVector.reshape(80))
        overflowProbLeft = h_left > 250
        overflowProbRight = h_right > 250

        if overflowProbLeft:
            self.prob_pi[0] = 1
            self.prob_pi[1] = 0

        elif overflowProbRight:
            self.prob_pi[0] = 0
            self.prob_pi[1] = 1

        else:
            self.prob_pi[0] = np.exp(h_left) / (np.exp(h_left) + np.exp(h_right))
            self.prob_pi[1] = np.exp(h_right) / (np.exp(h_left) + np.exp(h_right))

        return np.random.choice(2, 1, p=self.prob_pi)

    def get_greedy_action(self, obs) -> int:
        return np.argmax(self.prob_pi)

    def update(
            self,
            obs,
            action: int,
            reward: float,
            terminated: bool,
            next_obs,
    ):

        featureVector_next = self.get_featureVector(next_obs[0], next_obs[1], next_obs[2], next_obs[3])
        featureVector_curr = self.get_featureVector(obs[0], obs[1], obs[2], obs[3])
        reward_factor = 1
        if terminated:
            reward_factor = -10
        delta_t = reward_factor * reward + self.discount_factor * np.dot(
            self.stateValueWeight.reshape(80), featureVector_next.reshape(80) * (not terminated)) - np.dot(
            self.stateValueWeight.reshape(80), featureVector_curr.reshape(80))

        self.stateValueWeight = self.stateValueWeight + self.learning_rate_stateValue * delta_t * featureVector_curr

        if action == 0:
            self.policyWeightLeft = self.policyWeightLeft + self.learning_rate_policy * delta_t * featureVector_curr

        else:
            self.policyWeightRight = self.policyWeightRight + self.learning_rate_policy * delta_t * featureVector_curr

def episode_run (
    Agent,
    env,
    testing_training_case: str,
    ):
    # TBD
    observation, info = env.reset()
    terminated = False
    truncated = False

    while (not terminated) and (not truncated):
        if testing_training_case == "testing":
            action = Agent.get_greedy_action(observation)
        elif testing_training_case == "training":
            action = Agent.get_action(observation)  # agent policy that uses the observation and info
        
        else:
            return None
        
        next_observation, reward, terminated, truncated, info = env.step(action)
        Agent.update(observation, action, reward, terminated, next_observation)
        observation = next_observation

    return observation, reward,

env = gym.make("Drone2D-Uncertain-v0", render_sim = False, Sensor_noise_level = "none",
                   Actuator_noise_level = "none", Environmental_disturbance = "none")

# learning_rate = [0.01, 0.1, 1]
temperature = [0.1, 5, 50]

reward_train0 = [0,0,0]
reward_test0 = [0,0,0]

# for lr in range(len(learning_rate)):
for run in range(10):
    firstAgent = DroneActorCriticAgent(learning_rate_policy= 1/32, learning_rate_stateValue= 1/32, discount_factor = 0.95)
    for segment in range(500):
        for episode in range(10):
            # a episode
            observationTrain, rewardTrain= episode_run(firstAgent, env, "training")
            # if segment == 499:
            #     reward_train0[lr] += rewardTrain
    
        observationTest, rewardTest = episode_run(firstAgent, env, "testing")
        # if segment == 499:
        #     reward_test0[lr] += rewardTest
