import abc
import random
import copy


class LegalActionsAuthority(abc.ABC):
    """
    Abstract class that filters the legal actions in a state, among the actions set
    """
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def LegalActions(self, state):
        pass # return legal_actions_list

class AllActionsLegalAuthority(LegalActionsAuthority):  # Utility class that always allows all actions
    def __init__(self, actions_set):
        super().__init__()
        self.actions_set = actions_set

    def LegalActions(self, state):
        return self.actions_set


class Policy(abc.ABC):
    """
    Abstract class that selects an action from a state
    """
    def __init__(self, legal_actions_authority):
        super().__init__()
        self.legal_actions_authority = legal_actions_authority

    @abc.abstractmethod
    def Select(self, state):
        pass  # return action

class Random(Policy):  # Selects randomly one of the legal actions
    def __init__(self, legal_actions_authority):
        super().__init__(legal_actions_authority)

    def Select(self, state):
        legal_actions_set = self.legal_actions_authority.LegalActions(state)
        if len(legal_actions_set) == 0:
            return None
        return random.choice(list(legal_actions_set))

class EpsilonGreedy(Policy):  # Selects randomly with probability epsilon, otherwise selects the most valuable action,
                              # based on a static state evaluation.
                              # It uses a private playground environment.
    def __init__(self, state_to_value_dict, legal_actions_authority,
                 environment, epsilon=0.1, gamma=0.9, number_of_trials_per_action=1):
        super().__init__(legal_actions_authority)
        self.state_to_value_dict = copy.deepcopy(state_to_value_dict)  # To avoid unintentional interference
        self.environment = copy.deepcopy(environment)  # To avoid unintentional interference
        self.epsilon = epsilon
        self.gamma = gamma
        self.number_of_trials_per_action = max(number_of_trials_per_action, 1)

    def Select(self, state):
        legal_actions_set = self.legal_actions_authority.LegalActions(state)
        if len(legal_actions_set) == 0:
            return None
        random_0to1 = random.random()
        if random_0to1 < self.epsilon:  # Random choice
            return random.choice(list(legal_actions_set))
        # Greedy choice
        highest_value = float('-inf')
        selected_actions_list = []
        for candidate_action in legal_actions_set:
            average_value = 0
            for trialNdx in range(self.number_of_trials_per_action):
                self.environment.SetState(state)
                new_state, reward, done, info = self.environment.step(candidate_action)
                average_value += reward + self.gamma * self.state_to_value_dict[new_state]
            average_value = average_value/self.number_of_trials_per_action
            if average_value > highest_value:
                highest_value = average_value
                selected_actions_list = [candidate_action]
            elif average_value == highest_value:
                selected_actions_list.append(candidate_action)
        return random.choice(selected_actions_list)


class PolicyEvaluator:
    def __init__(self, environment, gamma=0.9, minimum_change=0.01,
                 number_of_selections_per_state=1,  # Should be 1 for deterministic policies
                 maximum_number_of_iterations=1000,
                 initial_value=0):
        """
        Implementation of policy evaluation, Cf. Reinforcement Learning, Sutton and Barto, p. 98
        It uses a private playground environment.
        """
        self.environment = copy.deepcopy(environment)  # Must implement StatesSet(), SetState(s)
        self.gamma = gamma  # The discount factor
        self.minimum_change = minimum_change  # Equivalent of theta in the book
        self.number_of_selections_per_state = number_of_selections_per_state
        self.maximum_number_of_iterations = maximum_number_of_iterations
        self.initial_value = initial_value

    def Evaluate(self, policy):
        minimum_change_is_achieved = True
        states_set = self.environment.StatesSet()
        state_to_value_dict = {s: self.initial_value for s in states_set}
        completed_iterations = 0
        while minimum_change_is_achieved and completed_iterations < self.maximum_number_of_iterations:
            change = 0
            for state in states_set:
                previous_value = state_to_value_dict[state]
                average_value = 0
                for selectionNdx in range(self.number_of_selections_per_state):
                    self.environment.SetState(state)
                    selected_action = policy.Select(state)
                    new_state, reward, done, info = self.environment.step(selected_action)
                    average_value += reward + self.gamma * state_to_value_dict[new_state]
                average_value = average_value/self.number_of_selections_per_state
                state_to_value_dict[state] = average_value
                change = max(change, abs(previous_value - average_value))
                #print ("improvement = {}".format(improvement))
            minimum_change_is_achieved = change >= self.minimum_change
            completed_iterations += 1
        return state_to_value_dict
