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
                 environment, epsilon=0.1, gamma=0.9, number_of_trials_per_action=1):
        super().__init__(legal_actions_authority)
        self.state_to_value_dict = copy.deepcopy(state_to_value_dict)  # To avoid unintentional interference
        self.environment = copy.deepcopy(environment)  # To avoid unintentional interference
        self.epsilon = epsilon
        self.gamma = gamma
        self.number_of_trials_per_action = max(number_of_trials_per_action, 1)  # Should be 1 for deterministic environments

    def ActionProbabilities(self, state):
        legal_actions_set = self.legal_actions_authority.LegalActions(state)
        if len(legal_actions_set) == 0:
            return {}
        action_to_probability_dict = {}
        highest_value = float('-inf')
        best_actions_list = []
        for candidate_action in legal_actions_set:
            average_value = 0
            for trialNdx in range(self.number_of_trials_per_action):
                self.environment.SetState(state)
                new_state, reward, done, info = self.environment.step(candidate_action)
                average_value += reward + self.gamma * self.state_to_value_dict[new_state]
            average_value = average_value / self.number_of_trials_per_action
            if average_value > highest_value:
                highest_value = average_value
                best_actions_list = [candidate_action]
            elif average_value == highest_value:
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
        #average_value_exploded = False
        while minimum_change_is_achieved and completed_iterations < self.maximum_number_of_iterations:
            change = 0
            updated_state_to_value_dict = copy.deepcopy(state_to_value_dict)
            for state in states_set:
                previous_value = state_to_value_dict[state]
                average_value = 0
                for selectionNdx in range(self.number_of_selections_per_state):  # More than 1, for non-deterministic policies
                    self.environment.SetState(state)
                    selected_action = policy.Select(state)
                    # The following call will eliminate the non-deterministic behavior of the environment
                    # because we don't call step()
                    new_state_to_probability_reward = self.environment.TransitionProbabilitiesAndRewards(selected_action)
                    average_value += sum(probability * (reward + self.gamma * state_to_value_dict[new_state]) for (new_state, (probability, reward) ) in new_state_to_probability_reward.items())
                average_value = average_value/self.number_of_selections_per_state
                updated_state_to_value_dict[state] = average_value
                change = max(change, abs(previous_value - average_value))
            state_to_value_dict = updated_state_to_value_dict
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
                 epsilon=0.1,
                 initial_value=0,
                 number_of_trials_per_action=100,
                 maximum_number_of_iterations=100,
                 print_steps=False):
        self.environment = copy.deepcopy(environment)
        self.policy_evaluator = policy_evaluator
        self.legal_actions_authority = legal_actions_authority
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_value = initial_value
        #self.number_of_trials_per_action = number_of_trials_per_action  # For deterministic environments, should be 1
        self.maximum_number_of_iterations = maximum_number_of_iterations
        self.print_steps = print_steps

    def IteratePolicy(self):
        states_set = self.environment.StatesSet()
        actions_set = self.environment.ActionsSet()
        iterated_policy = Greedy({s: list(actions_set)[0] for s in states_set}, self.legal_actions_authority)  # Initialize with the same action for each state
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
                    self.environment.SetState(state)
                    new_state_to_probability_reward = self.environment.TransitionProbabilitiesAndRewards(
                        candidate_action)
                    expected_value = sum(probability * (reward + self.gamma * state_to_updated_value_dict[new_state]) for (new_state, (probability, reward) ) in new_state_to_probability_reward.items())
                    action_to_expected_value_dict[candidate_action] = expected_value
                most_valuable_action = self.MostValuableAction(action_to_expected_value_dict)
                state_to_most_valuable_action[state] = most_valuable_action
            # Create a new greedy policy
            iterated_policy = Greedy(state_to_most_valuable_action, self.legal_actions_authority)

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
        for (action, value) in action_to_expected_value_dict.items():
            if value > highest_value:
                highest_value = value
                most_valuable_action = action
        return most_valuable_action