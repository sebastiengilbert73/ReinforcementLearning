import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random
import ReinforcementLearning.algorithms.policy as rl_policy
#import ReinforcementLearning.environments.dpenv as dpenv
import ReinforcementLearning.environments.attributes as env_attributes

class FrozenLake(env_attributes.DynamicProgramming, env_attributes.Episodic):
    def __init__(self, size='4x4'):
        super().__init__()
        if size == '4x4':
            self.frozen_lake = gym.make('FrozenLake-v0')
        elif size == '8x8':
            self.frozen_lake = gym.make('FrozenLake8x8-v0')
        else:
            raise NotImplementedError("FrozenLake.__init__(): Not implemented size {}".format(size))
        self.frozen_lake.reset()
        self.actions_set = set(range(4))
        self.states_set = set(range(self.frozen_lake.nrow * self.frozen_lake.ncol))
        self.holes = None
        if size == '4x4':
            self.holes = [(1, 1), (1, 3), (2, 3), (3, 0)]
        elif size == '8x8':
            self.holes = [(2, 3), (3, 5), (4, 3), (5, 1), (5, 2), (5, 6), (6, 1), (6, 4), (6, 6), (7, 3)]
        self.goals = None
        if size == '4x4':
            self.goals = [(3, 3)]
        elif size == '8x8':
            self.goals = [(7, 7)]


    def render(self):
        self.frozen_lake.render()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = 0
        self.frozen_lake.reset()
        return (self.state, 0, False, {})

    def close(self):
        pass

    def StatesSet(self):
        return self.states_set

    def ActionsSet(self):
        return self.actions_set

    def SetState(self, state):
        """if state not in self.states_set:
            raise ValueError("FrozenLake.SetState(): state {} is not part of the states set ({})".format(state, self.states_set))
        self.frozen_lake.reset()  # In case a previous experiment reached a terminal state
        self.frozen_lake.s = state
        self.state = state
        """
        raise NotImplementedError("FrozenLake.SetState(): Not allowed by openai gym interface.")

    def step(self, action):
        observation, reward, done, info = self.frozen_lake.step(action)
        self.state = self.frozen_lake.s
        return observation, reward, done, info

    def ComputeTransitionProbabilitiesAndRewards(self, state, action):
        newState_to_probabilityReward = {}
        """
        action: 0 = left     1 = down     2 = right       3 = up
        """
        action_to_probStateRewardDoneList = self.frozen_lake.P[state]
        probStateRewardDoneList = action_to_probStateRewardDoneList[action]
        for newStateNdx in range(len(probStateRewardDoneList)):
            newState = probStateRewardDoneList[newStateNdx][1]
            probability = probStateRewardDoneList[newStateNdx][0]
            reward = probStateRewardDoneList[newStateNdx][2]
            newState_to_probabilityReward[newState] = (probability, reward)
        return newState_to_probabilityReward


    def Coordinates(self, state):
        row = state//self.frozen_lake.ncol
        column = state - self.frozen_lake.ncol * row
        return (row, column)

    def StateFromCoordinates(self, row, column):
        return self.frozen_lake.ncol * row + column