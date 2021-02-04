import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random
import ReinforcementLearning.algorithms.policy as rl_policy
import ReinforcementLearning.environments.attributes as env_attributes

class Blackjack(env_attributes.Tabulatable,
                env_attributes.Episodic):
    def __init__(self):
        self.blackjack_env = gym.make('Blackjack-v0')
        self.reset()
        self.actions_set = set(range(2))  # 0 = stand;  1 = hit
        self.states_set = self.BuildStatesSet() #set(range(361))  # State 360 is terminal
            # player_sum (4-21): 18
            # dealer's card (1-10): 10
            # player holds a usable ace: 2
            # 18 x 10 x 2 + 1
        self.terminal_state = 360

    def reset(self):
        (player_sum, dealer_card, usable_ace) = self.blackjack_env.reset()
        #self.state = self.StateFromTuple((player_sum, dealer_card, usable_ace))
        self.state = (player_sum, usable_ace, dealer_card)
        done = False
        reward = 0
        if player_sum == 21:
            done = True
            reward = 1
        return (self.state, reward, done, {})

    """@staticmethod
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
    """

    def render(self):
        #print(self.TupleFromState(self.state))
        print(self.state)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        obs, reward, done, info = self.blackjack_env.step(action)
        #if done:
        #    self.state = self.terminal_state
        #else:
        #self.state = self.StateFromTuple(obs)
        (player_sum, dealer_card, usable_ace) = obs
        self.state = (player_sum, usable_ace, dealer_card)
        return (self.state, reward, done, info)

    def close(self):
        pass

    def StatesSet(self):
        return self.states_set

    def ActionsSet(self):
        return self.actions_set

    def BuildStatesSet(self):
        states_set = set()
        for sum in range(12, 22):  # 10 cases
            for usable_ace in range(0, 2):  # 2 cases
                has_useable_ace = (usable_ace == 1)
                for dealer_card in range(1, 11):  # 10 cases
                    states_set.add((sum, has_useable_ace, dealer_card))
        return states_set


class HitUpTo(rl_policy.Policy):
    def __init__(self, maximum_hit_value=19):
        super().__init__(rl_policy.AllActionsLegalAuthority(set(range(2))))
        self.maximum_hit_value = maximum_hit_value

    def ActionProbabilities(self, state):
        # return action_to_probability_dict
        (player_sum, usable_ace, dealer_card) = state
        if player_sum <= self.maximum_hit_value:
            return {0: 0, 1: 1}  # Hit
        else:
            return {0: 1, 1: 0}  # Stand


class BlackjackES(env_attributes.Tabulatable,
                  env_attributes.ExplorationStarts,
                  env_attributes.Episodic):
    def __init__(self):
        super(env_attributes.GymCompatible).__init__()
        super(env_attributes.Tabulatable).__init__()
        super(env_attributes.ExplorationStarts).__init__()
        super(env_attributes.Episodic).__init__()
        self.state = None
        self.reset()
        self.actions_set = set(range(2))  # 0 = stand;  1 = hit
        self.states_set = self.BuildStatesSet()

    def step(self, action):
        # return (observation, reward, done, info_dict)
        info_dict = {'dealer_cards': [self.state[2]], 'player_card': None}
        if action == 0:  # stand
            dealer_card1 = self.state[2]
            # Dealer gets a card
            dealer_card2 = random.randint(1, 13)
            info_dict['dealer_cards'].append(dealer_card2)
            if dealer_card2 >= 10:
                dealer_card2 = 10
            dealer_sum = dealer_card1 + dealer_card2
            dealer_has_usable_ace = False
            if dealer_card1 == 1 or dealer_card2 == 1:
                dealer_has_usable_ace = True
                dealer_sum += 10
            dealer_busted = False
            while not dealer_busted and dealer_sum <= 16:
                dealer_card = random.randint(1, 13)
                info_dict['dealer_cards'].append(dealer_card)
                if dealer_card >= 10:
                    dealer_card = 10
                if dealer_card == 1:
                    if dealer_sum > 10:
                        dealer_sum += 1
                    else:
                        dealer_sum += 11
                        dealer_has_usable_ace = True
                else:
                    dealer_sum += dealer_card

                if dealer_sum > 21:
                    if dealer_has_usable_ace:
                        dealer_sum -= 10
                        dealer_has_usable_ace = False
                    else:
                        dealer_busted = True
            reward = 0  # Draw
            if dealer_busted:
                reward = 1
            else:
                if self.state[0] > dealer_sum:
                    reward = 1
                elif self.state[0] < dealer_sum:
                    reward = -1
            self.state = (self.state[0], self.state[1], dealer_sum)
            return (self.state, reward, True, info_dict)
        elif action == 1:  # hit
            player_sum = self.state[0]
            if player_sum > 21:
                raise ValueError("BlackjackES.step(): player_sum ({}) > 21".format(player_sum))
            player_has_usable_ace = self.state[1]
            new_card = random.randint(1, 13)
            info_dict['player_card'] = new_card
            if new_card >= 10:
                new_card = 10
            done = False
            reward = 0
            if new_card == 1:
                if player_sum > 10:
                    player_sum += 1
                else:
                    player_sum += 11
                    player_has_usable_ace = True
            else:
                player_sum += new_card
            if player_sum > 21:
                if player_has_usable_ace:
                    player_sum -= 10
                    player_has_usable_ace = False
                else:
                    done = True
                    reward = -1
            self.state = (player_sum, player_has_usable_ace, self.state[2])
            return (self.state, reward, done, info_dict)
        else:
            raise ValueError("BlackjackES.step(): action = {}. It should be 0 for stand or 1 for hit".format(action))

    def reset(self):
        info_dict = {'player_cards': [], 'dealer_card': None}
        usable_ace = False
        card1 = random.randint(1, 13)
        info_dict['player_cards'].append(card1)
        if card1 >= 10:
            card1 = 10
        card2 = random.randint(1, 13)
        info_dict['player_cards'].append(card2)
        if card2 >= 10:
            card2 = 10
        sum = card1 + card2
        if card1 == 1 or card2 == 1:
            usable_ace = True
            sum += 10
        dealer_card = random.randint(1, 13)
        info_dict['dealer_card'] = [dealer_card]
        if dealer_card >= 10:
            dealer_card = 10
        self.state = (sum, usable_ace, dealer_card)
        done = False
        reward = 0
        if sum == 21:
            done = True
            # Check if the dealer has a natural as well:
            if dealer_card == 10:
                dealer_second_card = random.randint(1, 13)
                info_dict['dealer_card'].append(dealer_second_card)
                if dealer_second_card == 1:
                    reward = 0
            elif dealer_card == 1:
                dealer_second_card = random.randint(1, 13)
                info_dict['dealer_card'].append(dealer_second_card)
                if dealer_second_card >= 10:
                    reward = 0
            else:
                reward = 1
        return (self.state, reward, done, info_dict)

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

class BlackjackES_noFaces(env_attributes.Tabulatable,
                  env_attributes.ExplorationStarts,
                  env_attributes.Episodic):
    def __init__(self):
        super(env_attributes.GymCompatible).__init__()
        super(env_attributes.Tabulatable).__init__()
        super(env_attributes.ExplorationStarts).__init__()
        super(env_attributes.Episodic).__init__()
        self.state = None
        self.reset()
        self.actions_set = set(range(2))  # 0 = stand;  1 = hit
        self.states_set = self.BuildStatesSet()

    def step(self, action):
        # return (observation, reward, done, info_dict)
        info_dict = {'dealer_cards': [self.state[2]], 'player_card': None}
        if action == 0:  # stand
            dealer_card1 = self.state[2]
            # Dealer gets a card
            dealer_card2 = random.randint(1, 10)
            info_dict['dealer_cards'].append(dealer_card2)
            if dealer_card2 >= 10:
                dealer_card2 = 10
            dealer_sum = dealer_card1 + dealer_card2
            dealer_has_usable_ace = False
            if dealer_card1 == 1 or dealer_card2 == 1:
                dealer_has_usable_ace = True
                dealer_sum += 10
            dealer_busted = False
            while not dealer_busted and dealer_sum <= 16:
                dealer_card = random.randint(1, 10)
                info_dict['dealer_cards'].append(dealer_card)
                if dealer_card >= 10:
                    dealer_card = 10
                if dealer_card == 1:
                    if dealer_sum > 10:
                        dealer_sum += 1
                    else:
                        dealer_sum += 11
                        dealer_has_usable_ace = True
                else:
                    dealer_sum += dealer_card

                if dealer_sum > 21:
                    if dealer_has_usable_ace:
                        dealer_sum -= 10
                        dealer_has_usable_ace = False
                    else:
                        dealer_busted = True
            reward = 0  # Draw
            if dealer_busted:
                reward = 1
            else:
                if self.state[0] > dealer_sum:
                    reward = 1
                elif self.state[0] < dealer_sum:
                    reward = -1
            self.state = (self.state[0], self.state[1], dealer_sum)
            return (self.state, reward, True, info_dict)
        elif action == 1:  # hit
            player_sum = self.state[0]
            if player_sum > 21:
                raise ValueError("BlackjackES_noFaces.step(): player_sum ({}) > 21".format(player_sum))
            player_has_usable_ace = self.state[1]
            new_card = random.randint(1, 10)
            info_dict['player_card'] = new_card
            if new_card >= 10:
                new_card = 10
            done = False
            reward = 0
            if new_card == 1:
                if player_sum > 10:
                    player_sum += 1
                else:
                    player_sum += 11
                    player_has_usable_ace = True
            else:
                player_sum += new_card
            if player_sum > 21:
                if player_has_usable_ace:
                    player_sum -= 10
                    player_has_usable_ace = False
                else:
                    done = True
                    reward = -1
            self.state = (player_sum, player_has_usable_ace, self.state[2])
            return (self.state, reward, done, info_dict)
        else:
            raise ValueError("BlackjackES_noFaces.step(): action = {}. It should be 0 for stand or 1 for hit".format(action))

    def reset(self):
        info_dict = {'player_cards': [], 'dealer_card': None}
        usable_ace = False
        card1 = random.randint(1, 10)
        info_dict['player_cards'].append(card1)
        if card1 >= 10:
            card1 = 10
        card2 = random.randint(1, 10)
        info_dict['player_cards'].append(card2)
        if card2 >= 10:
            card2 = 10
        sum = card1 + card2
        if card1 == 1 or card2 == 1:
            usable_ace = True
            sum += 10
        dealer_card = random.randint(1, 10)
        info_dict['dealer_card'] = dealer_card
        if dealer_card >= 10:
            dealer_card = 10
        self.state = (sum, usable_ace, dealer_card)
        done = False
        reward = 0
        if sum == 21:
            done = True
            # Check if the dealer has a natural as well:
            if dealer_card == 10:
                dealer_second_card = random.randint(1, 10)
                info_dict['dealer_card'].append(dealer_second_card)
                if dealer_second_card == 1:
                    reward = 0
            elif dealer_card == 1:
                dealer_second_card = random.randint(1, 10)
                info_dict['dealer_card'].append(dealer_second_card)
                if dealer_second_card >= 10:
                    reward = 0
            else:
                reward = 1
        return (self.state, reward, done, info_dict)

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
            raise TypeError("BlackjackES_noFaces.SetState(): The passed object {} is not a tuple".format(sum_usableAce_dealerCard))
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

class BJSuttonBarto(rl_policy.Policy):
    # Cf. Reinforcement Learning, Sutton and Barto, p.121
    def __init__(self):
        self.state_to_action = {}
        for has_usable_ace in [True, False]:
            for player_sum in range(4, 22):
                for dealer_card in range(1, 11):
                    state = (player_sum, has_usable_ace, dealer_card)
                    if has_usable_ace:
                        if dealer_card == 1:
                            if player_sum < 19:
                                self.state_to_action[state] = 1
                            else:
                                self.state_to_action[state] = 0
                        elif dealer_card < 9:
                            if player_sum < 18:
                                self.state_to_action[state] = 1
                            else:
                                self.state_to_action[state] = 0
                        else:  # player_sum >= 9
                            if player_sum < 19:
                                self.state_to_action[state] = 1
                            else:
                                self.state_to_action[state] = 0
                    else:  # No usable ase
                        if dealer_card == 1:
                            if player_sum < 17:
                                self.state_to_action[state] = 1
                            else:
                                self.state_to_action[state] = 0
                        elif dealer_card < 4:
                            if player_sum < 13:
                                self.state_to_action[state] = 1
                            else:
                                self.state_to_action[state] = 0
                        elif dealer_card < 7:
                            if player_sum < 12:
                                self.state_to_action[state] = 1
                            else:
                                self.state_to_action[state] = 0
                        else:
                            if player_sum < 17:
                                self.state_to_action[state] = 1
                            else:
                                self.state_to_action[state] = 0

    def ActionProbabilities(self, state):
        action = self.state_to_action[state]
        return {action: 1}

class Iterated(rl_policy.Policy):
    def __init__(self):
        self.state_to_action = {}
        for has_usable_ace in [True, False]:
            for player_sum in range(4, 22):
                for dealer_card in range(1, 11):
                    state = (player_sum, has_usable_ace, dealer_card)
                    if has_usable_ace:
                        if player_sum < 19:
                            self.state_to_action[state] = 1
                        else:
                            self.state_to_action[state] = 0
                    else:  # No usable ase
                        if dealer_card == 1:
                            if player_sum < 16:
                                self.state_to_action[state] = 1
                            else:
                                self.state_to_action[state] = 0
                        elif dealer_card < 8:
                            if player_sum < 12:
                                self.state_to_action[state] = 1
                            else:
                                self.state_to_action[state] = 0
                        elif dealer_card < 9:
                            if player_sum < 13:
                                self.state_to_action[state] = 1
                            else:
                                self.state_to_action[state] = 0
                        elif dealer_card < 10:
                            if player_sum < 14:
                                self.state_to_action[state] = 1
                            else:
                                self.state_to_action[state] = 0
                        else:
                            if player_sum < 13:
                                self.state_to_action[state] = 1
                            else:
                                self.state_to_action[state] = 0

    def ActionProbabilities(self, state):
        action = self.state_to_action[state]
        return {action: 1}