import abc
import random

class Policy(abc.ABC):
    """ Abstract class that selects an action from a state """
    def __init__(self, legal_actions_authority):
        super().__init__()
        self.legal_actions_authority = legal_actions_authority

    @abc.abstractmethod
    def Select(self, state):
        pass  # return action


class LegalActionsAuthority(abc.ABC):
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

class Random(Policy):
    def __init__(self, legal_actions_authority):
        super().__init__(legal_actions_authority)

    def Select(self, state):
        legal_actions_set = self.legal_actions_authority.LegalActions(state)
        return random.choice(list(legal_actions_set))

class PolicyEvaluator:
    def __init__(self, environment, gamma=0.9, minimum_change=0.01,
                 number_of_selections_per_state=1,
                 maximum_number_of_iterations=1000,
                 initial_value=0):
        """Implementation of policy evaluation, Cf. Reinforcement Learning, Sutton and Barto, p. 98"""
        self.environment = environment  # Must implement StatesSet(), SetState(s)
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
