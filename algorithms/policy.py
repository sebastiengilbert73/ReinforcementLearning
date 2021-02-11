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

class EpsilonGreedy(Policy):
    """
    Selects the most valuable action with probability (1 - epsilon). Otherwise, randomly selects an action
    """
    def __init__(self, epsilon, stateAction_to_value):
        self.epsilon = epsilon
        self.stateAction_to_value = stateAction_to_value
        self.state_to_stateActions = {}  # Build in advance the dictionary of state to state-action pairs
        for ((state, action), value) in self.stateAction_to_value.items():
            if state in self.state_to_stateActions:
                self.state_to_stateActions[state].append((state, action))
            else:
                self.state_to_stateActions[state] = [(state, action)]

    def ActionProbabilities(self, state):
        stateActions_list = self.state_to_stateActions[state]
        if len(stateActions_list) == 0:
            return {}
        most_valuable_action = None
        highest_value = float('-inf')
        for (_state, action) in stateActions_list:
            value = self.stateAction_to_value[(_state, action)]
            if value > highest_value:
                highest_value = value
                most_valuable_action = action
        number_of_actions = len(stateActions_list)
        action_to_probability = {}
        for (_state, action) in stateActions_list:
            action_to_probability[action] = self.epsilon/number_of_actions
        action_to_probability[most_valuable_action] += (1.0 - self.epsilon)
        return action_to_probability