import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random
import ReinforcementLearning.algorithms.policy as rl_policy
import ReinforcementLearning.environments.dpenv as dpenv
import math


class JacksCarRental(dpenv.DynamicProgrammingEnv):
    # Cf. 'Reinforcement Learning', Sutton and Barto, p. 98
    metadata = {
        'render.modes': ['human']
    }
    def __init__(self, deterministic=False):
        super().__init__()
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
        self.deterministic = deterministic
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

    def step(self, action):
        # End of the day: Move cars
        number_of_cars_moved_from_location1_to_location2 = action  # [-5, -4, ..., 5]
        (cars_at_location1, cars_at_location2) = self.NumberOfCarsAtEachLocation(self.state)
        # Move the cars
        cars_at_location1 -= number_of_cars_moved_from_location1_to_location2
        cars_at_location2 += number_of_cars_moved_from_location1_to_location2
        if cars_at_location1 < 0:
            raise ValueError("JacksCarRental.step(): action = {}, self.state = {}; cars_at_location1 ({}) < 0".format(action, self.state, cars_at_location1))
        if cars_at_location2 < 0:
            raise ValueError(
                "JacksCarRental.step(): action = {}, self.state = {}; cars_at_location2 ({}) < 0".format(
                    action, self.state, cars_at_location2))
        if cars_at_location1 > 20:
            cars_at_location1 = 20
        if cars_at_location2 > 20:
            cars_at_location2 = 20

        if self.deterministic:
            # Returns
            location1_returns = self.location1_return_average
            cars_at_location1 = min(cars_at_location1 + location1_returns, 20)
            location2_returns = self.location2_return_average
            cars_at_location2 = min(cars_at_location2 + location2_returns, 20)

            # Rentals
            location1_rentals = min(self.location1_rental_average, cars_at_location1)
            cars_at_location1 -= location1_rentals
            location2_rentals = min(self.location2_rental_average, cars_at_location2)
            cars_at_location2 -= location2_rentals
        else:
            # Returns
            location1_returns = self.rng.poisson(self.location1_return_average)
            cars_at_location1 = min(cars_at_location1 + location1_returns, 20)
            location2_returns = self.rng.poisson(self.location2_return_average)
            cars_at_location2 = min(cars_at_location2 + location2_returns, 20)

            # Rentals
            location1_rentals = min(self.rng.poisson(self.location1_rental_average), cars_at_location1)
            cars_at_location1 -= location1_rentals
            location2_rentals = min(self.rng.poisson(self.location2_rental_average), cars_at_location2)
            cars_at_location2 -= location2_rentals

        reward = -self.cost_for_move * abs(number_of_cars_moved_from_location1_to_location2) \
            + self.rental_reward * (location1_rentals + location2_rentals)

        new_state = self.StateFromCarsAtEachLocation(cars_at_location1, cars_at_location2)
        self.state = new_state

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

    def TransitionProbabilitiesAndRewards(self, action):
        (cars_at_location1, cars_at_location2) = self.NumberOfCarsAtEachLocation(self.state)
        number_of_moves_from_location1_to_location2 = action  # [-5, -4, ...., 5]
        transition_probabilities_arr = np.zeros((21, 21), dtype=float)
        final_state_to_weighted_reward = {s: 0 for s in self.StatesSet()}

        poisson_maximum = 13
        for returns_at_location1 in range(poisson_maximum + 1):
            start_cars_at_location1 = np.clip(cars_at_location1 + returns_at_location1 - number_of_moves_from_location1_to_location2, 0, 20)
            for returns_at_location2 in range(poisson_maximum + 1):
                start_cars_at_location2 = np.clip(cars_at_location2 + returns_at_location2 + number_of_moves_from_location1_to_location2, 0, 20)
                for rentals_at_location1 in range(poisson_maximum + 1):
                    actual_rentals_at_location1 = min(rentals_at_location1, start_cars_at_location1)
                    final_cars_at_location1 = start_cars_at_location1 - actual_rentals_at_location1
                    for rentals_at_location2 in range(poisson_maximum + 1):
                        actual_rentals_at_location2 = min(rentals_at_location2, start_cars_at_location2)
                        final_cars_at_location2 = start_cars_at_location2 - actual_rentals_at_location2
                        probability = Poisson(self.location1_return_average, returns_at_location1) * \
                            Poisson(self.location1_rental_average, rentals_at_location1) * \
                            Poisson(self.location2_return_average, returns_at_location2) * \
                            Poisson(self.location2_rental_average, rentals_at_location2)
                        reward = -self.cost_for_move * abs(number_of_moves_from_location1_to_location2) + \
                            self.rental_reward * (actual_rentals_at_location1 + actual_rentals_at_location2)
                        if final_cars_at_location1 < 0 or final_cars_at_location1 > 20 or final_cars_at_location2 < 0 or final_cars_at_location2 > 20:
                            print(" ********** JacksCarRental.TransitionProbabilitiesAndRewards() **********")
                            print ("final_cars_at_location1 = {}; final_cars_at_location2 = {}".format(final_cars_at_location1, final_cars_at_location2))
                            print("self.state = {}; action = {}".format(self.state, action))
                            print("cars_at_location1 = {}; cars_at_location2 = {}".format(cars_at_location1, cars_at_location2))
                            jacks_possible_moves = JacksPossibleMoves()
                            legal_actions = jacks_possible_moves.LegalActions(self.state)
                            print ("legal_actions = {}".format(legal_actions))
                            print ("returns_at_location1 = {}".format(returns_at_location1))
                            print("start_cars_at_location1 = {}".format(start_cars_at_location1))
                            print("returns_at_location2 = {}".format(returns_at_location2))
                            print("start_cars_at_location2 = {}".format(start_cars_at_location2))
                            print("rentals_at_location1 = {}; actual_rentals_at_location1 = {}".format(rentals_at_location1, actual_rentals_at_location1))
                            print("rentals_at_location2 = {}; actual_rentals_at_location2 = {}".format(rentals_at_location2, actual_rentals_at_location2))
                            print("")
                        transition_probabilities_arr[final_cars_at_location1, final_cars_at_location2] += probability
                        final_state = self.StateFromCarsAtEachLocation(final_cars_at_location1, final_cars_at_location2)
                        final_state_to_weighted_reward[final_state] += probability * reward

        state_to_probability_and_reward = {}
        for new_state in self.StatesSet():
            (final_cars_at_location1, final_cars_at_location2) = self.NumberOfCarsAtEachLocation(new_state)
            transition_probability = transition_probabilities_arr[final_cars_at_location1, final_cars_at_location2]
            expected_reward = 0
            if transition_probability > 1e-9:
                expected_reward = final_state_to_weighted_reward[new_state] / transition_probability
            state_to_probability_and_reward[new_state] = (transition_probability, expected_reward)
        return state_to_probability_and_reward


class JacksPossibleMoves(rl_policy.LegalActionsAuthority):
    def __init__(self):
        super().__init__()

    def LegalActions(self, state):
        (cars_at_location1, cars_at_location2) = JacksCarRental.NumberOfCarsAtEachLocation(state)
        minimum = max(-5, -cars_at_location2)
        maximum = min(5, cars_at_location1)
        legal_actions_list = []
        for moves in range(minimum, maximum + 1):
            legal_actions_list.append(moves)

        return set(legal_actions_list)


def Poisson(lambda_, n):
    if n < 0:
        return 0
    return pow(lambda_, n) * math.exp(-lambda_)/math.factorial(n)

def ProbabilityOfDelta(returns_lambda, rentals_lambda, customers_delta, poisson_maximum=12):
    # customers_delta = returns - rentals
    probability = 0
    for returns in range(0, poisson_maximum + 1):
        rentals = returns - customers_delta
        if rentals >= 0 and rentals <= poisson_maximum:
            probability += Poisson(returns_lambda, returns) * Poisson(rentals_lambda, rentals)
    return probability

def ReturnsAndRentalsDistribution(returns_lambda, rentals_lambda, customers_delta, poisson_maximum=12):
    # customers_delta = returns - rentals
    probability_of_delta = ProbabilityOfDelta(returns_lambda, rentals_lambda, customers_delta)
    if probability_of_delta < 1e-9:
        return {}
    returns_rentals_to_probability = {}
    for returns in range(0, poisson_maximum + 1):
        rentals = returns - customers_delta
        probability = Poisson(returns_lambda, returns) * Poisson(rentals_lambda, rentals)/probability_of_delta
        returns_rentals_to_probability[(returns, rentals)] = probability
    return returns_rentals_to_probability

def ExpectedNumberOfRentals(returns_lambda, rentals_lambda, customers_delta, poisson_maximum=12):
    returns_and_rentals_distribution = ReturnsAndRentalsDistribution(returns_lambda, rentals_lambda, customers_delta, poisson_maximum)
    expected_rentals = 0
    for ((returns, rentals), probability) in returns_and_rentals_distribution.items():
        expected_rentals += rentals * probability
    return expected_rentals
