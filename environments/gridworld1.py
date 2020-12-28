import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class GridWorld1(gym.Env):  # Cf. ''Reinforcement Learning', Sutton and Barto, p.71
    # Class structure inspired by https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self):
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

    def Coordinates(self):
        if self.state < 0 or self.state > 24:
            raise ValueError("GridWorld1.Coordinates(): state {} is out of range [0, 24]".format(self.state))
        row = self.state // 5
        column = self.state - row * 5
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


class GridWorld2x2(gym.Env):  # Random policy, gamma=0.9: values: V0 = 7.316; V1 = 2.573; V2 = 2.573; V3 = 1.196
    # Class structure inspired by https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self):
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