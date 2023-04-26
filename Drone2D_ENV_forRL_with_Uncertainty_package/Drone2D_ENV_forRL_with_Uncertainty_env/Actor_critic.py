# actor-critic agent (for reference)

import numpy as np
import random
import matplotlib.pyplot as plt
import Drone2D_ENV_forRL_with_Uncertainty_env
import gymnasium as gym
import pygame

class cartpoleActorCriticAgent:
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

        self.policyWeightLeft = np.random.default_rng().uniform(-0.001, 0.001, (6, 10))
        self.policyWeightRight = np.random.default_rng().uniform(-0.001, 0.001, (6, 10))
        self.stateValueWeight = np.random.default_rng().uniform(-0.001, 0.001, (6, 10))
        self.policyWeight0 = np.random.default_rng().uniform(-0.001, 0.001)
        self.stateValueWeight0 = np.random.default_rng().uniform(-0.001, 0.001)

        self.learning_rate_stateValue = learning_rate_stateValue
        self.learning_rate_policy = learning_rate_policy
        self.discount_factor = discount_factor
        self.prob_pi = np.ones(2)

    def get_featureVector(self, velocity_x, velocity_y, angularV, angle, x, y):
        featureVector = np.full((6, 10), False, dtype=bool)
        cartPosition_x = x
        cartPosition_y = y
        cartVelocity_x = velocity_x
        cartVelocity_y = velocity_y
        poleAngle = angle
        poleAngularV = angularV
        pi = 3.1415926

        cartPosition_xBin = np.linspace(0, 1000, 10)
        cartPosition_yBin = np.linspace(0, 800, 10)
        cartVelocity_xBin = np.linspace(-10, 10, 10)
        cartVelocity_yBin = np.linspace(-10, 10, 10)
        poleAngleBin = np.linspace(-pi, pi, 10)
        poleAngularVBin = np.linspace(-pi, pi, 10)


        for index, bin in enumerate(cartVelocity_xBin):

            inFirstBin = cartVelocity_x <= bin and index == 0
            inBetweenBin = cartVelocity_x <= bin and cartVelocity_x > cartVelocity_xBin[index - 1]
            inLastBin = cartVelocity_x > cartVelocity_xBin[index - 1] and index == 9
            if (inFirstBin or inBetweenBin or inLastBin):
                featureVector[0][index] = True

        for index, bin in enumerate(cartVelocity_xBin):

            inFirstBin = cartVelocity <= bin and index == 0
            inBetweenBin = cartVelocity <= bin and cartVelocity > cartVelocityBin[index - 1]
            inLastBin = cartVelocity > cartVelocityBin[index - 1] and index == 9
            if (inFirstBin or inBetweenBin or inLastBin):
                featureVector[1][index] = True

        for index, bin in enumerate(poleAngleBin):

            inFirstBin = poleAngle <= bin and index == 0
            inBetweenBin = poleAngle <= bin and poleAngle > poleAngleBin[index - 1]
            inLastBin = poleAngle > poleAngleBin[index - 1] and index == 9
            if (inFirstBin or inBetweenBin or inLastBin):
                featureVector[2][index] = True

        for index, bin in enumerate(poleAngularVBin):

            inFirstBin = poleAngularV <= bin and index == 0
            inBetweenBin = poleAngularV <= bin and poleAngularV > poleAngularVBin[index - 1]
            inLastBin = poleAngularV > poleAngularVBin[index - 1] and index == 9
            if (inFirstBin or inBetweenBin or inLastBin):
                featureVector[3][index] = True

        for index, bin in enumerate(cartPosition_xBin):

            inFirstBin = cartPosition_x <= bin and index == 0
            inBetweenBin = cartPosition_x <= bin and cartPosition_x > cartPosition_xBin[index - 1]
            inLastBin = cartPosition_x > cartPosition_xBin[index - 1] and index == 9
            if (inFirstBin or inBetweenBin or inLastBin):
                featureVector[0][index] = True

        return featureVector

    def get_action(self, obs) -> int:
        # Boltzmann policy
        featureVector = self.get_featureVector(obs[0], obs[1], obs[2], obs[3])
        h_left = self.policyWeight0 + np.dot(
            self.policyWeightLeft.reshape(40), featureVector.reshape(40))
        h_right = self.policyWeight0 + np.dot(
            self.policyWeightRight.reshape(40), featureVector.reshape(40))
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
            self.stateValueWeight.reshape(40), featureVector_next.reshape(40) * (not terminated)) - np.dot(
            self.stateValueWeight.reshape(40), featureVector_curr.reshape(40))

        self.stateValueWeight = self.stateValueWeight + self.learning_rate_stateValue * delta_t * featureVector_curr

        if action == 0:
            self.policyWeightLeft = self.policyWeightLeft + self.learning_rate_policy * delta_t * featureVector_curr

        else:
            self.policyWeightRight = self.policyWeightRight + self.learning_rate_policy * delta_t * featureVector_curr