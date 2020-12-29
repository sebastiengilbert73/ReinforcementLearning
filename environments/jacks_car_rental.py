import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random
import ReinforcementLearning.algorithms.policy as rl_policy


class JacksCarRental(gym.Env):
    # Cf. 'Reinforcement Learning', Sutton and Barto, p. 98
    metadata = {
        'render.modes': ['human']
    }
    def __init__(self):
        self.observation_space = spaces.Discrete(21 * 21)  # ([0, 1, ... 20], [0, 1, ... 20])
        self.actions_list = list(range(-5, 6))  # [-5, -4, ..., 4, 5]
        self.action_space = spaces.Discrete(len(self.actions_list))
        self.state = 0
        self.seed()
        self.rental_reward = 10.
        self.cost_for_move = 2.
        self.location1_rental_average = 3
        self.location2_rental_average = 4
        self.location1_return_average = 3
        self.location2_return_average = 2
        self.rng = np.random.default_rng()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @staticmethod
    def NumberOfCarsAtEachLocation(state):
        if state < 0 or state >= 441:
            raise ValueError("JacksCarRental.NumberOfCarsAtEachLocation(): state {} is out of range [0, 440]".format(state))
        cars_at_location1 = state // 21
        cars_at_location2 = state - 21 * cars_at_location1
        return (cars_at_location1, cars_at_location2)

    def StateFromCarsAtEachLocation(self, cars_at_location1, cars_at_location2):
        return 21 * cars_at_location1 + cars_at_location2

    def step(self, action_index):
        # End of the day: Move cars
        number_of_cars_moved_from_location1_to_location2 = action_index - 5  # [-5, -4, ..., 5]
        (cars_at_location1, cars_at_location2) = self.NumberOfCarsAtEachLocation()
        # Move the cars
        cars_at_location1 -= number_of_cars_moved_from_location1_to_location2
        cars_at_location2 += number_of_cars_moved_from_location1_to_location2
        if cars_at_location1 < 0:
            raise ValueError("JacksCarRental.step(): action_index = {}, self.state = {}; cars_at_location1 ({}) < 0".format(action_index, self.state, cars_at_location1))
        if cars_at_location2 < 0:
            raise ValueError(
                "JacksCarRental.step(): action_index = {}, self.state = {}; cars_at_location2 ({}) < 0".format(
                    action_index, self.state, cars_at_location2))
        if cars_at_location1 > 20:
            cars_at_location1 = 20
        if cars_at_location2 > 20:
            cars_at_location2 = 20

        # Returns
        location1_returns = self.rng.poisson(self.location1_return_average)
        cars_at_location1 = max(cars_at_location1 + location1_returns, 20)
        location2_returns = self.rng.poisson(self.location2_return_average)
        cars_at_location2 = max(cars_at_location2 + location2_returns, 20)

        # Rentals
        location1_rentals = max(self.rng.poisson(self.location1_rental_average), cars_at_location1)
        cars_at_location1 -= location1_rentals
        location2_rentals = max(self.rng.poisson(self.location2_rental_average), cars_at_location2)
        cars_at_location2 -= location2_rentals

        reward = -self.cost_for_move * abs(number_of_cars_moved_from_location1_to_location2) \
            + self.rental_reward * (location1_rentals + location2_rentals)

        new_state = self.StateFromCarsAtEachLocation(cars_at_location1, cars_at_location2)

        return (new_state, reward, False, None)

    def reset(self):
        self.state = np.random.randint(self.observation_space.n)
        return self.state

    def render(self, mode='human'):
        (cars_at_location1, cars_at_location2) = self.NumberOfCarsAtEachLocation()
        print ("({}, {})".format(cars_at_location1, cars_at_location2))

    def close(self):
        pass

    def StatesSet(self):  # To be used by PolicyEvaluator
        return set(range(self.observation_space.n))

    def SetState(self, index):
        if index < 0 or index >= self.observation_space.n:
            raise ValueError("JacksCarRental.SetState(): index {} is out of range [0, {}}]".format(index, self.observation_space.n - 1))
        self.state = index

    def ActionsSet(self):
        return set(self.actions_list)


class JacksPossibleMoves(rl_policy.LegalActionsAuthority):
    def __init__(self):
        super().__init__()

    def LegalActions(self, state):
        (cars_at_location1, cars_at_location2) = JacksCarRental.NumberOfCarsAtEachLocation(state)
        minimum = max(-5, -cars_at_location2)
        maximum = min(5, cars_at_location1)
        return set(list(range(minimum, maximum + 1)))