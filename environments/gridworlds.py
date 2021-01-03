import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random
import ReinforcementLearning.algorithms.policy as rl_policy
import ReinforcementLearning.environments.dpenv as dpenv


class GridWorld1(dpenv.DynamicProgrammingEnv):  # Cf. 'Reinforcement Learning', Sutton and Barto, p.71
    # Class structure inspired by https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(25)
        self.action_space = spaces.Discrete(4)
        self.actionsList = ['north', 'south', 'east', 'west']
        self.state = 0
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action_index):
        action = self.actionsList[action_index]
        (row, column) = self.Coordinates()
        if row == 0 and column == 1:  # A -> A'
            self.state = self.StateFromCoordinates(4, 1)
            return (self.state, 10., False, None)
        elif row == 0 and column == 3:  # B -> B'
            self.state = self.StateFromCoordinates(2, 3)
            return (self.state, 5., False, None)
        elif row == 0 and action == 'north':
            return (self.state, -1., False, None)
        elif row == 4 and action == 'south':
            return (self.state, -1., False, None)
        elif column == 0 and action == 'west':
            return (self.state, -1., False, None)
        elif column == 4 and action == 'east':
            return (self.state, -1., False, None)
        else:
            new_row = row
            new_column = column
            if action == 'north':
                new_row -= 1
            elif action == 'south':
                new_row += 1
            elif action == 'east':
                new_column += 1
            elif action == 'west':
                new_column -= 1
            new_state = self.StateFromCoordinates(new_row, new_column)
            self.state = new_state
            return (self.state, 0, False, None)

    def reset(self):
        self.state = np.random.randint(25)
        return self.state

    def render(self, mode='human'):
        coordinates = self.Coordinates()
        for row in range(5):
            for column in range(5):
                if row == coordinates[0] and column == coordinates[1]:
                    print (' X ', end='', flush=True)
                else:
                    print (' . ', end='', flush=True)
            print()

    def close(self):
        pass

    def Coordinates(self, state=None):
        if state is None:
            state = self.state
        if state < 0 or state > 24:
            raise ValueError("GridWorld1.Coordinates(): state {} is out of range [0, 24]".format(state))
        row = state // 5
        column = state - row * 5
        return (row, column)

    def StateFromCoordinates(self, row, column):
        if row < 0 or row > 4 or column < 0 or column > 4:
            raise ValueError("GridWorld1.StateFromCoordinates(): Coordinates ({}, {}) are out of range".format(row, column))
        return row * 5 + column

    def StatesSet(self):  # To be used by PolicyEvaluator
        return set(range(25))

    def SetState(self, index):
        if index < 0 or index > 24:
            raise ValueError("GridWorld1.SetState(): index {} is out of range [0, 24]".format(index))
        self.state = index

    def ActionsSet(self):
        return set(range(4))

    def ComputeTransitionProbabilitiesAndRewards(self, action):
        new_state_to_probability_reward = {}
        origin_coordinates = self.Coordinates()

        for new_state in self.StatesSet():
            new_coordinates = self.Coordinates(new_state)
            probability = 0
            reward = 0
            if origin_coordinates[0] == 0 and origin_coordinates[1] == 1:
                if new_coordinates[0] == 4 and new_coordinates[1] == 1:  # No matter what action:
                    probability = 1
                    reward = 10
                else:
                    probability = 0
                    reward = 0
            elif origin_coordinates[0] == 0 and origin_coordinates[1] == 3:
                if new_coordinates[0] == 2 and new_coordinates[1] == 3:  # No matter what action:
                    probability = 1
                    reward = 5
                else:
                    probability = 0
                    reward = 0
            elif new_coordinates[0] - origin_coordinates[0] == 1 and \
                    new_coordinates[1] == origin_coordinates[1] and action == 1:
                probability = 1
            elif new_coordinates[0] - origin_coordinates[0] == -1 and \
                    new_coordinates[1] == origin_coordinates[1] and action == 0:
                probability = 1
            elif new_coordinates[1] - origin_coordinates[1] == 1 and \
                new_coordinates[0] == origin_coordinates[0] and action == 2:
                probability = 1
            elif new_coordinates[1] - origin_coordinates[1] == -1 and \
                new_coordinates[0] == origin_coordinates[0] and action == 3:
                probability = 1
            elif origin_coordinates[0] == 0 and new_coordinates[0] == 0 and \
                origin_coordinates[1] == new_coordinates[1] and action == 0:
                probability = 1
                reward = -1
            elif origin_coordinates[0] == 4 and new_coordinates[0] == 4 and \
                origin_coordinates[1] == new_coordinates[1] and action == 1:
                probability = 1
                reward = -1
            elif origin_coordinates[1] == 0 and new_coordinates[1] == 0 and \
                origin_coordinates[0] == new_coordinates[0] and action == 3:
                probability = 1
                reward = -1
            elif origin_coordinates[1] == 4 and new_coordinates[1] == 4 and \
                origin_coordinates[0] == new_coordinates[0] and action == 2:
                probability = 1
                reward = -1
            new_state_to_probability_reward[new_state] = (probability, reward)
        return new_state_to_probability_reward

class GridWorld2x2(dpenv.DynamicProgrammingEnv):
    # Class structure inspired by https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
    # Random policy, gamma=0.9:
    #       final exact values:     V0 = 7.316; V1 = 2.573; V2 = 2.573; V3 = 1.196
    # Epsilon-greedy:
    #   gamma=0.9, epsilon=0.1, initial values: V0 = V1 = V2 = V3
    #       final exact values:     V0 = 14.912, V1 = 11.013, V2 = 11.013, V3 = 9.808
    #   gamma=0.9, epsilon=0, initial values: V0 = V1 = V2 = V3
    #       final exact values:     V0 = 15.658, V1 = 11.842, V2 = 11.842, V3 = 10.658
    # Optimal policy pi*:
    #       final exact values:     V0* = 26.316, V1* = 23.684, V2* = 23.684, V3* = 21.316
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(4)
        self.action_space = spaces.Discrete(4)
        self.actionsList = ['north', 'south', 'east', 'west']
        self.state = 0
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action_index):
        action = self.actionsList[action_index]
        (row, column) = self.Coordinates()
        if row == 0 and column == 0:
            self.state = self.StateFromCoordinates(1, 0)
            return (self.state, 5., False, None)
        elif row == 0 and action == 'north':
            return (self.state, -1., False, None)
        elif row == 1 and action == 'south':
            return (self.state, -1., False, None)
        elif column == 0 and action == 'west':
            return (self.state, -1., False, None)
        elif column == 1 and action == 'east':
            return (self.state, -1., False, None)
        else:
            new_row = row
            new_column = column
            if action == 'north':
                new_row -= 1
            elif action == 'south':
                new_row += 1
            elif action == 'east':
                new_column += 1
            elif action == 'west':
                new_column -= 1
            new_state = self.StateFromCoordinates(new_row, new_column)
            self.state = new_state
            return (self.state, 0, False, None)

    def reset(self):
        self.state = np.random.randint(4)
        return self.state

    def render(self, mode='human'):
        coordinates = self.Coordinates()
        for row in range(2):
            for column in range(2):
                if row == coordinates[0] and column == coordinates[1]:
                    print (' X ', end='', flush=True)
                else:
                    print (' . ', end='', flush=True)
            print()

    def close(self):
        pass

    def Coordinates(self):
        if self.state < 0 or self.state > 3:
            raise ValueError("GridWorld2x2.Coordinates(): state {} is out of range [0, 3]".format(self.state))
        row = self.state // 2
        column = self.state - row * 2
        return (row, column)

    def StateFromCoordinates(self, row, column):
        if row < 0 or row > 1 or column < 0 or column > 1:
            raise ValueError("GridWorld2x2.StateFromCoordinates(): Coordinates ({}, {}) are out of range".format(row, column))
        return row * 2 + column

    def StatesSet(self):  # To be used by PolicyEvaluator
        return set(range(4))

    def SetState(self, index):
        if index < 0 or index > 3:
            raise ValueError("GridWorld2x2.SetState(): index {} is out of range [0, 3]".format(index))
        self.state = index

    def ActionsSet(self):
        return set(range(4))

    def ComputeTransitionProbabilitiesAndRewards(self, action):
        new_state_to_probability_reward = {}
        for new_state in self.StatesSet():
            probability = 0
            reward = 0
            if new_state == 0:
                if self.state == 1 and action == 3 or \
                    self.state == 2 and action == 0:
                    probability = 1
            elif new_state == 1:
                if self.state == 3 and action == 0:
                    probability = 1
                if self.state == 1 and action == 0 or \
                    self.state == 1 and action == 2:
                    probability = 1
                    reward = -1
            elif new_state == 2:
                if self.state == 0:
                    probability = 1
                    reward = 5
                elif self.state == 2 and action == 3 or \
                    self.state == 2 and action == 1:
                    probability = 1
                    reward = -1
                elif self.state == 3 and action == 3:
                    probability = 1
            elif new_state == 3:
                if self.state == 1 and action == 1:
                    probability = 1
                elif self.state == 2 and action == 2:
                    probability = 1
                elif self.state == 3 and action == 2 or \
                    self.state == 3 and action == 1:
                    probability = 1
                    reward = -1
            new_state_to_probability_reward[new_state] = (probability, reward)
        return new_state_to_probability_reward




""" 
Optimal policies, to match 'Reinforcement Learning', Sutton and Barto, p. 79
"""
class GridWorld1OptimalPolicy(rl_policy.Policy):
    def __init__(self):
        legal_actions_authority = rl_policy.AllActionsLegalAuthority(set(range(4)))
        super().__init__(legal_actions_authority)

    def ActionProbabilities(self, state):
        if state == 0:
            return {0: 0, 1: 0, 2: 1, 3: 0}
        elif state in [1, 3]:
            return {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        elif state in [2, 4, 8, 9]:
            return {0: 0, 1: 0, 2: 0, 3: 1}
        elif state in [5, 10, 15, 20]:
            return {0: 0.5, 1: 0, 2: 0.5, 3: 0}
        elif state in [6, 11, 16, 21]:
            return {0: 1, 1: 0, 2: 0, 3: 0}
        elif state in [7, 12, 13, 14, 17, 18, 19, 22, 23, 24]:
            return {0: 0.5, 1: 0, 2: 0, 3: 0.5}
        else:
            raise ValueError("GridWorld1OptimalPolicy.ActionProbabilities(): State {} out of range".format(state))

class GridWorld2x2OptimalPolicy(rl_policy.Policy):
    def __init__(self):
        legal_actions_authority = rl_policy.AllActionsLegalAuthority(set(range(4)))
        super().__init__(legal_actions_authority)

    def ActionProbabilities(self, state):
        action_to_probability_dict = {}
        actions_set = self.legal_actions_authority.LegalActions(state)
        if state == 0:
            return {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        elif state == 1:
            return {0: 0, 1: 0, 2: 0, 3: 1}
        elif state == 2:
            return {0: 1, 1: 0, 2: 0, 3: 0}
        elif state == 3:
            return {0: 0.5, 1: 0, 2: 0, 3: 0.5}
        else:
            raise ValueError("GridWorld2x2OptimalPolicy.ActionProbabilities(): state {} is out of range [0, 3]".format(state))
