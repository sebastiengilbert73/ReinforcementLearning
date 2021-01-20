import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random
import ReinforcementLearning.algorithms.policy as rl_policy
import ReinforcementLearning.environments.attributes as env_attributes

class Blackjack(env_attributes.GymCompatible,
                env_attributes.Tabulatable):
    def __init__(self):
        self.blackjack_env = gym.make('Blackjack-v0')
        self.reset()
        self.actions_set = set(range(2))  # 0 = stand;  1 = hit
        self.states_set = set(range(361))  # State 360 is terminal
            # player_sum (4-21): 18
            # dealer's card (1-10): 10
            # player holds a usable ace: 2
            # 18 x 10 x 2 + 1
        self.terminal_state = 360

    def reset(self):
        (player_sum, dealer_card, usable_ace) = self.blackjack_env.reset()
        self.state = self.StateFromTuple((player_sum, dealer_card, usable_ace))
        done = False
        reward = 0
        if player_sum == 21:
            done = True
            reward = 1
        return (self.state, reward, done, {})

    @staticmethod
    def StateFromTuple(playerSum_dealerCard_usableAce):
        (player_sum, dealer_card, usable_ace) = playerSum_dealerCard_usableAce
        state = (player_sum - 4) * 20 + (dealer_card - 1) * 2
        if usable_ace:
            state += 1
        return state

    @staticmethod
    def TupleFromState(state):
        usable_ace = state % 2
        player_sum = 4 + state // 20
        dealer_card = 1 + (state - (player_sum - 4) * 20 - usable_ace) // 2
        return (player_sum, dealer_card, usable_ace == 1)

    def render(self):
        print(self.TupleFromState(self.state))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        obs, reward, done, info = self.blackjack_env.step(action)
        #if done:
        #    self.state = self.terminal_state
        #else:
        self.state = self.StateFromTuple(obs)
        return (self.state, reward, done, info)

    def close(self):
        pass

    def StatesSet(self):
        return self.states_set

    def ActionsSet(self):
        return self.actions_set


class HitUpTo(rl_policy.Policy):
    def __init__(self, maximum_hit_value=19):
        super().__init__(rl_policy.AllActionsLegalAuthority(set(range(2))))
        self.maximum_hit_value = maximum_hit_value

    def ActionProbabilities(self, state):
        # return action_to_probability_dict
        (player_sum, dealer_card, usable_ace) = Blackjack.TupleFromState(state)
        if player_sum <= self.maximum_hit_value:
            return {0: 0, 1: 1}  # Hit
        else:
            return {0: 1, 1: 0}  # Stand


class BlackjackES(env_attributes.GymCompatible,
                  env_attributes.Tabulatable,
                  env_attributes.ExplorationStarts):
    def __init__(self):
        super(env_attributes.GymCompatible).__init__()
        super(env_attributes.Tabulatable).__init__()
        super(env_attributes.ExplorationStarts).__init__()
        self.state = None
        self.reset()
        self.actions_set = set(range(2))  # 0 = stand;  1 = hit
        self.states_set = self.BuildStatesSet()

    def step(self, action):
        # return (observation, reward, done, info_dict)
        if action == 0:  # stand
            pass

    def reset(self):
        usable_ace = False
        card1 = random.randint(1, 11)
        card2 = random.randint(1, 11)
        if card1 == 11 or card2 == 11:
            usable_ace = True
        sum = card1 + card2
        if sum == 22:  # two aces
            sum = 12

        dealer_card = random.randint(1, 11)
        self.state = (sum, usable_ace, dealer_card)
        done = False
        reward = 0
        if sum == 21:
            done = True
            reward = 1
        return (self.state, reward, done, {})

    def render(self, mode):
        print(self.state)

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def ActionsSet(self):
        return self.actions_set

    def SetState(self, sum_usableAce_dealerCard):
        if not isinstance(sum_usableAce_dealerCard, tuple):
            raise TypeError("BlackjackES.SetState(): The passed object {} is not a tuple".format(sum_usableAce_dealerCard))
        if len(sum_usableAce_dealerCard) != 3:
            raise ValueError("The length of the passed tuple object ({}) is not 3".format(len(sum_usableAce_dealerCard)))

        self.state = sum_usableAce_dealerCard
        done = False
        reward = 0
        if sum_usableAce_dealerCard[0] == 21 and sum_usableAce_dealerCard[1] == True:
            done = True
            reward = 1
        if sum_usableAce_dealerCard[0] > 21:
            done = True
        return (self.state, reward, done, {})

    def BuildStatesSet(self):
        states_set = set()
        for sum in range(12, 22):  # 10 cases
            for usable_ace in range(0, 2):  # 2 cases
                has_useable_ace = (usable_ace == 1)
                for dealer_card in range(1, 11):  # 10 cases
                    states_set.add((sum, has_useable_ace, dealer_card))
        return states_set

    def StatesSet(self):
        return self.states_set