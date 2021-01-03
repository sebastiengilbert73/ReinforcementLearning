import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random
import ReinforcementLearning.algorithms.policy as rl_policy
import ReinforcementLearning.environments.dpenv as dpenv
import math

class GamblersProblem(dpenv.DynamicProgrammingEnv):
    def __init__(self, heads_probability=0.5):
        super().__init__()
        self.heads_probability = heads_probability
        self.state = 1
        self.actions_set = set(list(range(1, 100)))  # [1, 2, ..., 99]
        self.states_set = set(list(range(1, 100)))
        self.seed()
        self.rng = np.random.default_rng()

    def step(self, action):
        # return (observation, reward, done, info_dict)
        if action > self.state:
            raise ValueError("GamblersProblem.step(): action ({}) > self.state ({})".format(action, self.state))
        stake = action
        reward = 0
        done = False
        info_dict = None
        random_0to1 = random.random()
        if random_0to1 < self.heads_probability:  # gambler wins
            self.state += stake
            if self.state >= 100:
                self.state = 100
                reward = 1
                done = True
        else:  # gambler loses
            self.state -= stake
            if self.state <= 0:
                self.state = 0
                reward = 0
                done = True
        return (self.state, reward, done, info_dict)

    def reset(self):
        self.state = 1

    def render(self, mode):
        print ("capital = {}".format(self.state))

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def StatesSet(self):
        return self.states_set

    def SetState(self, state):
        if state not in self.states_set:
            raise ValueError("GamblersProblem.SetStates(): set_state ({}) is not in the states_set ({})".format(state, self.states_set))
        self.state = state

    def ActionsSet(self):
        return self.actions_set

    def ComputeTransitionProbabilitiesAndRewards(self, action):
        newState_to_probabilityAndReward_dict = {}
        stake = action
        for new_state in self.StatesSet():
            probability = 0
            reward = 0
            #if self.state - stake > 0 and self.state + stake < 100:  # No need to worry about end of episode
            if new_state == self.state + stake:  # A win
                probability = self.heads_probability
                if new_state == 100:
                    reward = 1
            elif new_state == self.state - stake:  # A loss
                probability = 1 - self.heads_probability
            newState_to_probabilityAndReward_dict[new_state] = (probability, reward)
        return newState_to_probabilityAndReward_dict


class GamblersPossibleStakes(rl_policy.LegalActionsAuthority):
    def __init__(self):
        super().__init__()

    def LegalActions(self, state):
        max_stake = state
        if state + max_stake > 100:
            max_stake = 100 - state
        return set(list(range(1, max_stake + 1)))