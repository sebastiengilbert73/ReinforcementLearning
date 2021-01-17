import random
import copy
import ReinforcementLearning.algorithms.policy as rl_policy
import ReinforcementLearning.environments.attributes as env_attributes

class PolicyEvaluator:
    def __init__(self, environment,
                 gamma=0.9,
                 minimum_change=0.01,
                 number_of_selections_per_state=1,  # Should be 1 for deterministic policies
                 maximum_number_of_iterations=1000,
                 initial_value=0):
        """
        Implementation of policy evaluation, Cf. Reinforcement Learning, Sutton and Barto, p. 98
        It uses a private playground environment.
        """
        if not isinstance(environment, env_attributes.DynamicProgramming):
            raise TypeError("PolicyEvaluator.__init__(): The environment type ({}) is not an instance of ReinforcementLearning.environments.attributes.DynamicProgramming".format(type(environment)))
        self.environment = copy.deepcopy(environment)  # Must implement StatesSet(), TransitionProbabilitiesAndRewards()
        self.gamma = gamma  # The discount factor
        self.minimum_change = minimum_change  # Equivalent of theta in the book
        self.number_of_selections_per_state = number_of_selections_per_state
        self.maximum_number_of_iterations = maximum_number_of_iterations
        self.initial_value = initial_value

    def Evaluate(self, policy):
        minimum_change_is_achieved = True
        states_set = self.environment.StatesSet()
        state_to_value_dict = {s: self.initial_value for s in states_set}
        #state_to_value_dict = {s: random.random() for s in states_set}
        completed_iterations = 0
        #average_value_exploded = False
        while minimum_change_is_achieved and completed_iterations < self.maximum_number_of_iterations:
            change = 0
            updated_state_to_value_dict = copy.deepcopy(state_to_value_dict)
            for state in states_set:
                previous_value = state_to_value_dict[state]
                average_value = 0
                for selectionNdx in range(self.number_of_selections_per_state):  # More than 1, for non-deterministic policies
                    #self.environment.SetState(state)
                    selected_action = policy.Select(state)
                    # The following call will eliminate the non-deterministic behavior of the environment
                    # because we don't call step()
                    new_state_to_probability_reward = self.environment.TransitionProbabilitiesAndRewards(state, selected_action)
                    average_value += sum(probability * (reward + self.gamma * state_to_value_dict[new_state]) for (new_state, (probability, reward) ) in new_state_to_probability_reward.items())
                average_value = average_value/self.number_of_selections_per_state
                updated_state_to_value_dict[state] = average_value
                change = max(change, abs(previous_value - average_value))
            state_to_value_dict = copy.deepcopy(updated_state_to_value_dict)
            minimum_change_is_achieved = change >= self.minimum_change
            completed_iterations += 1
        return (state_to_value_dict, change, completed_iterations)


class PolicyIterator:
    """
    Implementation of policy iteration, Cf. 'Reinforcement Learning', Sutton and Barto, p. 98
    """
    def __init__(self, environment,
                 policy_evaluator,
                 legal_actions_authority,
                 gamma=0.9,
                 maximum_number_of_iterations=100,
                 print_steps=False):
        if not isinstance(environment, env_attributes.DynamicProgramming):
            raise TypeError("PolicyIterator.__init__(): The environment type ({}) is not an instance of ReinforcementLearning.environments.attributes.DynamicProgramming".format(type(environment)))
        self.environment = copy.deepcopy(environment)
        self.policy_evaluator = policy_evaluator
        self.legal_actions_authority = legal_actions_authority
        self.gamma = gamma
        self.maximum_number_of_iterations = maximum_number_of_iterations
        self.print_steps = print_steps

    def IteratePolicy(self):
        states_set = self.environment.StatesSet()
        actions_set = self.environment.ActionsSet()
        iterated_policy = rl_policy.Greedy({s: random.choice(list(actions_set)) for s in states_set}, self.legal_actions_authority)
        last_policy_state_to_most_valuable_action = iterated_policy.state_to_most_valuable_action

        policy_is_stable = False
        completed_iterations = 0
        while not policy_is_stable and completed_iterations < self.maximum_number_of_iterations:
            (state_to_updated_value_dict, last_change_magnitude, evaluation_completed_iterations) = self.policy_evaluator.Evaluate(iterated_policy)
            if self.print_steps:
                print("Evaluation {} completed.".format(completed_iterations + 1))
                print("last_change_magnitude = {}; evaluation_completed_iterations = {}".format(last_change_magnitude, evaluation_completed_iterations))
            state_to_most_valuable_action = {}
            for state in states_set:
                action_to_expected_value_dict = {}
                legal_actions = self.legal_actions_authority.LegalActions(state)
                for candidate_action in legal_actions:
                    #self.environment.SetState(state)
                    new_state_to_probability_reward = self.environment.TransitionProbabilitiesAndRewards(state,
                        candidate_action)
                    expected_value = sum(probability * (reward + self.gamma * state_to_updated_value_dict[new_state]) for (new_state, (probability, reward) ) in new_state_to_probability_reward.items())
                    action_to_expected_value_dict[candidate_action] = expected_value
                most_valuable_action = self.MostValuableAction(action_to_expected_value_dict)
                state_to_most_valuable_action[state] = most_valuable_action
            # Create a new greedy policy
            iterated_policy = rl_policy.Greedy(state_to_most_valuable_action, self.legal_actions_authority)

            number_of_disagreements = 0
            for s in states_set:
                if iterated_policy.state_to_most_valuable_action[s] != last_policy_state_to_most_valuable_action[s]:
                    number_of_disagreements += 1
            if number_of_disagreements == 0:
                policy_is_stable = True

            if self.print_steps:
                print ("policy_is_stable = {}; number_of_disagreements = {}".format(policy_is_stable, number_of_disagreements))
            last_policy_state_to_most_valuable_action = iterated_policy.state_to_most_valuable_action
            completed_iterations += 1

        iterated_policy_state_to_actions_probabilities = {}
        for s in states_set:
            iterated_policy_state_to_actions_probabilities[s] = iterated_policy.ActionProbabilities(s)
        return (iterated_policy, iterated_policy_state_to_actions_probabilities, last_policy_state_to_most_valuable_action)


    @staticmethod
    def MostValuableAction(action_to_expected_value_dict):
        highest_value = float('-inf')
        most_valuable_action = None
        actions_list = list(action_to_expected_value_dict.keys())
        actions_list.sort()
        for action in actions_list:
            value = action_to_expected_value_dict[action]
            if value > highest_value:
                highest_value = value
                most_valuable_action = action
        return most_valuable_action

class ValueIterator:
    def __init__(self,
                 environment,
                 legal_actions_authority,
                 gamma,
                 minimum_change,
                 maximum_number_of_iterations):
        if not isinstance(environment, env_attributes.DynamicProgramming):
            raise TypeError("ValueIterator.__init__(): The environment type ({}) is not an instance of ReinforcementLearning.environments.attributes.DynamicProgramming".format(type(environment)))
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

class EpsilonGreedy(rl_policy.Policy):
        # Selects randomly with probability epsilon, otherwise selects the most valuable action,
        # based on a static state evaluation.
        # It uses a private playground environment.


    def __init__(self, state_to_value_dict, legal_actions_authority,
                 environment, epsilon=0.1, gamma=0.9):
        super().__init__(legal_actions_authority)
        if not isinstance(environment, env_attributes.TransitionDynamics):
            raise TypeError("EpsilonGreedy.__init__(): The environment type ({}) is not an instance of ReinforcementLearning.environments.attributes.TransitionDynamics".format(type(environment)))
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
                action_to_probability_dict[action] = self.epsilon / len(legal_actions_set) + (1 - self.epsilon) / len(
                    best_actions_list)
            else:
                action_to_probability_dict[action] = self.epsilon / len(legal_actions_set)
        return action_to_probability_dict