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


        self.shape = 20
        self.full = 8 * self.shape

        self.policyWeightLeftupThrust = np.random.default_rng().uniform(-0.001, 0.001, (8,self.shape))
        self.policyWeightLeftmaintainThrust = np.random.default_rng().uniform(-0.001, 0.001, (8,self.shape))
        self.policyWeightLeftdownThrust = np.random.default_rng().uniform(-0.001, 0.001, (8,self.shape))
        self.policyWeightRightupThrust = np.random.default_rng().uniform(-0.001, 0.001, (8,self.shape))
        self.policyWeightRightmaintainThrust = np.random.default_rng().uniform(-0.001, 0.001, (8,self.shape))
        self.policyWeightRightdownThrust = np.random.default_rng().uniform(-0.001, 0.001, (8,self.shape))

        self.stateValueWeight = np.random.default_rng().uniform(-0.001, 0.001, (8,self.shape))
        self.policyWeight0 = np.random.default_rng().uniform(-0.001, 0.001)
        self.stateValueWeight0 = np.random.default_rng().uniform(-0.001, 0.001)

        self.learning_rate_stateValue = learning_rate_stateValue
        self.learning_rate_policy = learning_rate_policy
        self.discount_factor = discount_factor
        self.prob_pi_left = np.ones(3)
        self.prob_pi_right = np.ones(3)


    def get_featureVector(self, obs):
        velocity_x, velocity_y, angularV, angle, x, y, left_thrust, right_thrust = obs
        featureVector = np.full((8, self.shape), False, dtype=bool)
        pi = 3.1415926

        dronePosition_xBin = np.linspace(0, 1000, self.shape)
        dronePosition_yBin = np.linspace(0, 800, self.shape)
        droneVelocity_xBin = np.linspace(-10, 10, self.shape)
        droneVelocity_yBin = np.linspace(-10, 10, self.shape)

        droneAngleBin = np.linspace(-pi, pi, self.shape)
        droneAngularVBin = np.linspace(-pi, pi, self.shape)
        DroneLeftThrust = np.linspace(200, 1500, self.shape)
        DroneRightThrust = np.linspace(200, 1500, self.shape)

        


        for index, bin in enumerate(dronePosition_xBin):

            inFirstBin = x <= bin and index == 0
            inBetweenBin = x <= bin and x > dronePosition_xBin[index - 1]
            inLastBin = x > dronePosition_xBin[index - 1] and index == self.shape - 1
            if (inFirstBin or inBetweenBin or inLastBin):
                featureVector[0][index] = True

        for index, bin in enumerate(dronePosition_yBin):

            inFirstBin = y <= bin and index == 0
            inBetweenBin = y <= bin and y > dronePosition_yBin[index - 1]
            inLastBin = y > dronePosition_yBin[index - 1] and index == self.shape - 1
            if (inFirstBin or inBetweenBin or inLastBin):
                featureVector[1][index] = True

        for index, bin in enumerate(droneVelocity_xBin):

            inFirstBin = velocity_x <= bin and index == 0
            inBetweenBin = velocity_x <= bin and velocity_x > droneVelocity_xBin[index - 1]
            inLastBin = velocity_x > droneVelocity_xBin[index - 1] and index == self.shape - 1
            if (inFirstBin or inBetweenBin or inLastBin):
                featureVector[2][index] = True

        for index, bin in enumerate(droneVelocity_yBin):

            inFirstBin = velocity_y <= bin and index == 0
            inBetweenBin = velocity_y <= bin and velocity_y > droneVelocity_yBin[index - 1]
            inLastBin = velocity_y > droneVelocity_yBin[index - 1] and index == self.shape - 1
            if (inFirstBin or inBetweenBin or inLastBin):
                featureVector[3][index] = True

        for index, bin in enumerate(droneAngularVBin):

            inFirstBin = angularV <= bin and index == 0
            inBetweenBin = angularV <= bin and angularV > droneAngularVBin[index - 1]
            inLastBin = angularV > droneAngularVBin[index - 1] and index == self.shape - 1
            if (inFirstBin or inBetweenBin or inLastBin):
                featureVector[4][index] = True

        for index, bin in enumerate(droneAngleBin):

            inFirstBin = angle <= bin and index == 0
            inBetweenBin = angle <= bin and angle > droneAngleBin[index - 1]
            inLastBin = angle > droneAngleBin[index - 1] and index == self.shape - 1 
            if (inFirstBin or inBetweenBin or inLastBin):
                featureVector[5][index] = True

        for index, bin in enumerate(DroneLeftThrust):

            inFirstBin = left_thrust <= bin and index == 0
            inBetweenBin = left_thrust <= bin and left_thrust > DroneLeftThrust[index - 1]
            inLastBin = left_thrust > DroneLeftThrust[index - 1] and index == self.shape - 1
            if (inFirstBin or inBetweenBin or inLastBin):
                featureVector[6][index] = True

        for index, bin in enumerate(DroneRightThrust):

            inFirstBin = right_thrust <= bin and index == 0
            inBetweenBin = right_thrust <= bin and right_thrust > DroneRightThrust[index - 1]
            inLastBin = right_thrust > DroneRightThrust[index - 1] and index == self.shape - 1
            if (inFirstBin or inBetweenBin or inLastBin):
                featureVector[7][index] = True

        return featureVector

    def get_action(self, obs) -> int:
        # Boltzmann policy
        featureVector = self.get_featureVector(obs)
        h_leftUpthrust = self.policyWeight0 + np.dot(
            self.policyWeightLeftupThrust.reshape(self.full), featureVector.reshape(self.full))

        h_leftKeepthrust = self.policyWeight0 + np.dot(
            self.policyWeightLeftmaintainThrust.reshape(self.full), featureVector.reshape(self.full))
        
        h_leftDownthrust = self.policyWeight0 + np.dot(
            self.policyWeightLeftdownThrust.reshape(self.full), featureVector.reshape(self.full))
        
        denom_left = (np.exp(h_leftUpthrust) + np.exp(h_leftKeepthrust) + np.exp(h_leftDownthrust))

        self.prob_pi_left[0] = np.exp(h_leftUpthrust) / denom_left
        self.prob_pi_left[1] = np.exp(h_leftKeepthrust) / denom_left
        self.prob_pi_left[2] = np.exp(h_leftDownthrust) / denom_left

        h_rightUpthrust = self.policyWeight0 + np.dot(
            self.policyWeightRightupThrust.reshape(self.full), featureVector.reshape(self.full))

        h_rightKeepthrust = self.policyWeight0 + np.dot(
            self.policyWeightRightmaintainThrust.reshape(self.full), featureVector.reshape(self.full))
        
        h_rightDownthrust = self.policyWeight0 + np.dot(
            self.policyWeightRightdownThrust.reshape(self.full), featureVector.reshape(self.full))
        
        denom_right = (np.exp(h_rightUpthrust) + np.exp(h_rightKeepthrust) + np.exp(h_rightDownthrust))

        self.prob_pi_right[0] = np.exp(h_rightUpthrust) / denom_right
        self.prob_pi_right[1] = np.exp(h_rightKeepthrust) / denom_right
        self.prob_pi_right[2] = np.exp(h_rightDownthrust) / denom_right

        return [np.random.choice(3, 1, p=self.prob_pi_left), np.random.choice(3, 1, p=self.prob_pi_right)]

    def get_greedy_action(self, obs) -> int:
        return [np.argmax(self.prob_pi_left), np.argmax(self.prob_pi_right)]

    def update(
            self,
            obs,
            action,
            reward: float,
            terminated: bool,
            next_obs,
    ):

        featureVector_next = self.get_featureVector(next_obs)
        featureVector_curr = self.get_featureVector(obs)

        delta_t = reward + (not terminated) * self.discount_factor * np.dot(
            self.stateValueWeight.reshape(self.full), featureVector_next.reshape(self.full)) - np.dot(
            self.stateValueWeight.reshape(self.full), featureVector_curr.reshape(self.full))

        self.stateValueWeight = self.stateValueWeight + self.learning_rate_stateValue * delta_t * featureVector_curr

        left_thrust_up = action[0] == 0
        left_thrust_maintain = action[0] == 1
        left_thrust_down = action[0] == 2

        right_thrust_up = action[1] == 0
        right_thrust_maintain = action[1] == 1
        right_thrust_down = action[1] == 2

        if left_thrust_up:
            self.policyWeightLeftupThrust = self.policyWeightLeftupThrust + self.learning_rate_policy * delta_t * featureVector_curr
        
        elif left_thrust_maintain:
            self.policyWeightLeftmaintainThrust = self.policyWeightLeftmaintainThrust + self.learning_rate_policy * delta_t * featureVector_curr

        elif left_thrust_down:
            self.policyWeightLeftdownThrust = self.policyWeightLeftdownThrust + self.learning_rate_policy * delta_t * featureVector_curr

        if right_thrust_up:
            self.policyWeightRightupThrust = self.policyWeightRightupThrust + self.learning_rate_policy * delta_t * featureVector_curr

        elif right_thrust_maintain:
            self.policyWeightRightmaintainThrust = self.policyWeightRightmaintainThrust + self.learning_rate_policy * delta_t * featureVector_curr
        
        elif right_thrust_down:
            self.policyWeightRightdownThrust = self.policyWeightRightdownThrust + self.learning_rate_policy * delta_t * featureVector_curr

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

testingenv = gym.make("Drone2D-Uncertain-v0", render_sim = True, Sensor_noise_level = "none",
                   Actuator_noise_level = "none", Environmental_disturbance = "none")

# learning_rate = [0.01, 0.1, 1]
temperature = [0.1, 5, 50]

reward_train0 = [0,0,0]
reward_test0 = [0,0,0]

# for lr in range(len(learning_rate)):
for run in range(1):
    firstAgent = DroneActorCriticAgent(learning_rate_policy= 1/32, learning_rate_stateValue= 1/32, discount_factor = 0.95)
    for segment in range(50):
        for episode in range(100):
            # a episode
            observationTrain, rewardTrain= episode_run(firstAgent, env, "training")
            # if segment == 499:
            #     reward_train0[lr] += rewardTrain
    
        observationTest, rewardTest = episode_run(firstAgent, testingenv, "testing")
        # if segment == 499:
        #     reward_test0[lr] += rewardTest
