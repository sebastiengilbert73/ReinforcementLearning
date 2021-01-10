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
        pass # return legal_actions_set

class AllActionsLegalAuthority(LegalActionsAuthority):
    """
    Utility class that always allows all actions
    """
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
    def ActionProbabilities(self, state):
        pass  # return action_to_probability_dict

    def Probability(self, state, action):
        action_to_probability_dict = self.ActionProbabilities(state)
        if action in action_to_probability_dict:
            return action_to_probability_dict[action]
        else:
            return 0

    def Select(self, state):
        action_to_probability_dict = self.ActionProbabilities(state)
        action_running_sum_list = []
        running_sum = 0
        for action, probability in action_to_probability_dict.items():
            running_sum += probability
            action_running_sum_list.append((action, running_sum))
        random_0to1 = random.random()
        for action_running_sum in action_running_sum_list:
            if action_running_sum[1] >= random_0to1:
                return action_running_sum[0]
        raise ValueError("Policy.Select(): Reached the end of the loop without returning. state = {}; action_running_sum_list = {}; random_0to1 = {}".format(state, action_running_sum_list, random_0to1))

class Random(Policy):  # Selects randomly one of the legal actions
    def __init__(self, legal_actions_authority):
        super().__init__(legal_actions_authority)

    def ActionProbabilities(self, state):
        legal_actions_set = self.legal_actions_authority.LegalActions(state)
        action_to_probability_dict = {}
        for action in legal_actions_set:
            action_to_probability_dict[action] = 1/len(legal_actions_set)
        return action_to_probability_dict

class EpsilonGreedy(Policy):  # Selects randomly with probability epsilon, otherwise selects the most valuable action,
                              # based on a static state evaluation.
                              # It uses a private playground environment.
    def __init__(self, state_to_value_dict, legal_actions_authority,
                 environment, epsilon=0.1, gamma=0.9):
        super().__init__(legal_actions_authority)
        self.state_to_value_dict = copy.deepcopy(state_to_value_dict)  # To avoid unintentional interference
        self.environment = copy.deepcopy(environment)  # To avoid unintentional interference
        self.epsilon = epsilon
        self.gamma = gamma

    def ActionProbabilities(self, state):
        legal_actions_set = self.legal_actions_authority.LegalActions(state)
        if len(legal_actions_set) == 0:
            return {}
        action_to_probability_dict = {}
        highest_value = float('-inf')
        best_actions_list = []
        for candidate_action in legal_actions_set:
            newState_to_probabilityReward = self.environment.TransitionProbabilitiesAndRewards(
                state, candidate_action)
            candidate_value = sum(newState_to_probabilityReward[new_state][0] * (
                newState_to_probabilityReward[new_state][1] + self.gamma * self.state_to_value_dict[new_state]
            ) for new_state in newState_to_probabilityReward)
            if candidate_value > highest_value:
                highest_value = candidate_action
                best_actions_list = [candidate_action]
            elif candidate_value == highest_value:
                best_actions_list.append(candidate_action)
        for action in legal_actions_set:
            if action in best_actions_list:
                action_to_probability_dict[action] = self.epsilon / len(legal_actions_set) + (1 - self.epsilon) / len(best_actions_list)
            else:
                action_to_probability_dict[action] = self.epsilon / len(legal_actions_set)
        return action_to_probability_dict

class Greedy(Policy):
    """
    Always selects the most valuable action, as kept in a table
    """
    def __init__(self, state_to_most_valuable_action, legal_actions_authority):
        super().__init__(legal_actions_authority)
        self.state_to_most_valuable_action = copy.deepcopy(state_to_most_valuable_action)

    def ActionProbabilities(self, state):
        legal_actions_set = self.legal_actions_authority.LegalActions(state)
        if self.state_to_most_valuable_action[state] not in legal_actions_set:  # Initialization error: Attribute an arbitrary legal action
            self.state_to_most_valuable_action[state] = list(legal_actions_set)[0]
        return {self.state_to_most_valuable_action[state]: 1}

