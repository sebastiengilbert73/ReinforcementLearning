import random
import copy
import ReinforcementLearning.algorithms.policy as rl_policy

class ValueIterator:
    def __init__(self,
                 environment,
                 legal_actions_authority,
                 gamma,
                 minimum_change,
                 maximum_number_of_iterations):
        self.environment = copy.deepcopy(environment)
        self.legal_actions_authority = legal_actions_authority
        self.gamma = gamma
        self.minimum_change = minimum_change
        self.maximum_number_of_iterations = maximum_number_of_iterations

    def Iterate(self):
        states_set = self.environment.StatesSet()
        actions_set = self.environment.ActionsSet()
        state_to_value = {s: 0 for s in states_set}
        evaluation_is_stable = False
        completed_iterations = 0
        while not evaluation_is_stable and completed_iterations < self.maximum_number_of_iterations:
            evaluation_is_stable = True
            state_to_most_valuable_action = {s: None for s in states_set}
            for origin_state in states_set:
                previous_value = state_to_value[origin_state]
                max_value = float('-inf')
                most_valuable_action = None
                for candidate_action in self.legal_actions_authority.LegalActions(origin_state):
                    #self.environment.SetState(origin_state)
                    newState_to_probabilityReward = self.environment.TransitionProbabilitiesAndRewards(origin_state, candidate_action)
                    candidate_value = sum(newState_to_probabilityReward.get(new_state, (0, 0))[0] *
                                          (newState_to_probabilityReward.get(new_state, (0, 0))[1] +
                                           self.gamma * state_to_value[new_state]) for new_state in states_set)
                    if candidate_value > max_value:
                        max_value = candidate_value
                        most_valuable_action = candidate_action

                state_to_value[origin_state] = max_value
                state_to_most_valuable_action[origin_state] = most_valuable_action
                delta_value = abs(max_value - previous_value)
                if delta_value > self.minimum_change:
                    evaluation_is_stable = False
            completed_iterations += 1
        # Return a greedy deterministic policy
        return rl_policy.Greedy(state_to_most_valuable_action, self.legal_actions_authority)