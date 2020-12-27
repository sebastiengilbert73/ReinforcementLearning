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
    def __init__(self, actions_list):
        super().__init__()
        self.actions_list = actions_list

    def LegalActions(self, state):
        return self.actions_list

class Random(Policy):
    def __init__(self, legal_actions_authority):
        super().__init__(legal_actions_authority)

    def Select(self, state):
        legal_actions_list = self.legal_actions_authority.LegalActions(state)
        return random.choice(legal_actions_list)

class PolicyEvaluator:
    def __init__(self, environment, gamma=0.9, minimum_improvement=0.01,
                 number_of_selections_per_state=1):
        """Implementation of policy evaluation, Cf. Reinforcement Learning, Sutton and Barto, p. 98"""
        self.environment = environment  # Must implement StatesSet(), SetState(s)
        self.gamma = gamma  # The discount factor
        self.minimum_improvement = minimum_improvement  # Equivalent of theta in the book
        self.number_of_selections_per_state = number_of_selections_per_state

    def Evaluate(self, policy, initial_value=0, maximum_number_of_iterations=1000):
        minimum_improvement_is_achieved = True
        states_set = self.environment.StatesSet()
        state_to_value_dict = {s: initial_value for s in states_set}
        completed_iterations = 0
        while minimum_improvement_is_achieved and completed_iterations < maximum_number_of_iterations:
            improvement = 0
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
                improvement = max(improvement, abs(previous_value - state_to_value_dict[state]))
                #print ("improvement = {}".format(improvement))
            minimum_improvement_is_achieved = improvement >= self.minimum_improvement
            completed_iterations += 1
        return state_to_value_dict
