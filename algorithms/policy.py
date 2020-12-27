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