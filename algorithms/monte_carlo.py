import random
import copy
import statistics
import ReinforcementLearning.environments.attributes as env_attributes
import ReinforcementLearning.algorithms.policy as rl_policy
import ReinforcementLearning.utilities.rewards as rewards
import numpy as np


class FirstVisitPolicyEvaluator:
    """
    Implements algorithm in 'Reinforcement Learning', Sutton and Barto, p.113
    Environment interface:
        StatesSet()
        reset() -> observation, reward, done, info
        state
        step(action) -> observation, reward, done, info
        Episode(policy, maximum_number_of_steps) -> observationActionReward_list
    Policy interface:
        Select(state)
    """
    def __init__(self, environment,
                 gamma=0.9,
                 number_of_iterations=1000,
                 initial_value=0,
                 episode_maximum_length=1000):
        if not isinstance(environment, env_attributes.Tabulatable):
            raise TypeError("FirstVisitPolicyEvaluator.__init__(): The environment type ({}) is not an instance of ReinforcementLearning.environments.attributes.Tabulatable".format(type(environment)))
        if not isinstance(environment, env_attributes.GymCompatible):
            raise TypeError("FirstVisitPolicyEvaluator.__init__(): The environment type ({}) is not an instance of ReinforcementLearning.environments.attributes.GymCompatible".format(type(environment)))
        if not isinstance(environment, env_attributes.Episodic):
            raise TypeError(
                "FirstVisitPolicyEvaluator.__init__(): The environment type ({}) is not an instance of ReinforcementLearning.environments.attributes.Episodic".format(
                    type(environment)))
        self.environment = copy.deepcopy(environment)
        self.gamma = gamma
        self.number_of_iterations = number_of_iterations
        self.initial_value = initial_value
        self.episode_maximum_length = episode_maximum_length

    def Evaluate(self, policy, print_iteration=False):
        states_set = self.environment.StatesSet()
        state_to_value_dict = {s: self.initial_value for s in states_set}
        state_to_returns_dict = {s: [] for s in states_set}

        if print_iteration:
            print ("FirstVisitPolicyEvaluator.Evaluate()")

        for iteration in range(self.number_of_iterations):
            # Generate an episode
            observationActionReward_list = self.environment.Episode(policy, maximum_number_of_steps=self.episode_maximum_length)
            # [reward0, obs0, action1, reward1, obs1, action2, reward2, obs2, ...]
            observations_list = [observationActionReward_list[ndx] for ndx in
                                 range(len(observationActionReward_list)) if ndx%3 == 1]
            rewards_list = [observationActionReward_list[ndx] for ndx in
                            range(len(observationActionReward_list)) if ndx%3 == 0]
            observation_first_visit_is_encountered = {o: False for o in observations_list}
            for observationNdx in range(len(observations_list)):
                observation = observations_list[observationNdx]
                if observation in states_set and not observation_first_visit_is_encountered[observation]:
                    observationReturn = rewards.Return(rewards_list[observationNdx:], self.gamma)
                    state_to_returns_dict[observation].append(observationReturn)
                    observation_first_visit_is_encountered[observation] = True
                    state_to_value_dict[observation] = statistics.mean(state_to_returns_dict[observation])
            if print_iteration and iteration % 100 == 1:
                print('.', end='', flush=True)
        if print_iteration:
            print()
        return state_to_value_dict

class MonteCarloESPolicyIterator:
    """
    Implements algorithm 'Monte Carlo ES', in Reinforcement Learning, Sutton and Barto, p. 120
    Environment interface:
        Tabulatable
        Episodic
        ExplorationStarts
    """
    def __init__(self,
                 environment,
                 legal_actions_authority,
                 gamma=0.9,
                 number_of_iterations=1000,
                 initial_value=0,
                 episode_maximum_length=1000
                 ):
        if not isinstance(environment, env_attributes.Tabulatable):
            raise TypeError("MonteCarloESPolicyIterator.__init__(): The environment {} is not an instance of ReinforcementLearning.environments.attributes.Tabulatable".format(environment))
        if not isinstance(environment, env_attributes.Episodic):
            raise TypeError(
                "MonteCarloESPolicyIterator.__init__(): The environment {} is not an instance of ReinforcementLearning.environments.attributes.Episodic".format(environment))
        if not isinstance(environment, env_attributes.ExplorationStarts):
            raise TypeError(
                "MonteCarloESPolicyIterator.__init__(): The environment {} is not an instance of ReinforcementLearning.environments.attributes.ExplorationStarts".format(environment))
        if not isinstance(legal_actions_authority, rl_policy.LegalActionsAuthority):
            raise TypeError(
                "MonteCarloESPolicyIterator.__init__(): The legal_actions_authority {} is not an instance of ReinforcementLearning.algorithms.LegalActionsAuthority".format(legal_actions_authority))
        self.environment = copy.deepcopy(environment)
        self.legal_actions_authority = legal_actions_authority
        self.gamma = gamma
        self.number_of_iterations = number_of_iterations
        self.initial_value = initial_value
        self.episode_maximum_length = episode_maximum_length

    def IteratePolicy(self):
        states_set = self.environment.StatesSet()
        actions_set = self.environment.ActionsSet()
        # Initialization
        stateAction_to_value = {}
        stateAction_to_returns = {}
        state_to_mostValuableAction = {}
        for state in states_set:
            legal_actions_set = self.legal_actions_authority.LegalActions(state)
            state_to_mostValuableAction[state] = random.choice(list(legal_actions_set))
            for action in actions_set:
                stateAction_to_value[(state, action)] = self.initial_value
                stateAction_to_returns[(state, action)] = rewards.RunningSum()
        policy = rl_policy.Greedy(state_to_mostValuableAction, self.legal_actions_authority)

        for iteration in range(self.number_of_iterations):
            for start_state in states_set:
                start_legal_actions = self.legal_actions_authority.LegalActions(start_state)
                for start_legal_action in start_legal_actions:
                    episode = self.environment.Episode(policy,
                                                       start_state=start_state,
                                                       start_action=start_legal_action,
                                                       maximum_number_of_steps=self.episode_maximum_length)
                    stateAction_pairs = self.environment.StateActionPairs(episode)
                    rewards_list = [episode[ndx] for ndx in
                                    range(len(episode)) if ndx % 3 == 0]
                    rewards_list = rewards_list[1:]  # Get rid of reward 0, as it is not the consequence of an action
                    if len(stateAction_pairs) != len(rewards_list):
                        raise ValueError("MonteCarloESPolicyIterator.IteratePolicy(): len(stateAction_pairs) ({}) != len(rewards_list) ({})".format(len(stateAction_pairs), len(rewards_list)))
                    stateAction_is_encountered = {stateAction: False for stateAction in stateAction_pairs}
                    for stateActionNdx in range(len(stateAction_pairs)):
                        stateAction = stateAction_pairs[stateActionNdx]
                        if not stateAction_is_encountered[stateAction]:
                            return_value = rewards.Return(rewards_list[stateActionNdx:], self.gamma)
                            stateAction_to_returns[stateAction].Append(return_value)
                            stateAction_to_value[stateAction] = stateAction_to_returns[stateAction].Average()
                            stateAction_is_encountered[stateAction] = True
                    # Update policy through state_to_mostValuableAction
                    visited_states_list = [s for (s, a) in stateAction_pairs]
                    for visited_state in visited_states_list:
                        legal_actions = self.legal_actions_authority.LegalActions(visited_state)
                        highest_value = float('-inf')
                        most_valuable_action = None
                        for action in legal_actions:
                            action_value = stateAction_to_value[(visited_state, action)]
                            if action_value > highest_value:
                                highest_value = action_value
                                most_valuable_action = action
                        state_to_mostValuableAction[visited_state] = most_valuable_action
                    policy = rl_policy.Greedy(state_to_mostValuableAction, self.legal_actions_authority)
        return policy


class MonteCarloOnPolicyIterator():
    """
    Implements 'An epsilon-soft on-policy Monte Carlo control algorithm', 'Reinforcement Learning', Sutton & Barto p. 125
    Gets rid on the dependence on exploration starts
    Environment interface:
        Tabulatable
        Episodic
    """
    def __init__(self,
                 environment,
                 legal_actions_authority,
                 epsilon=0.1,
                 gamma=0.9,
                 number_of_iterations=1000,
                 episode_maximum_length=1000):
        if not isinstance(environment, env_attributes.Tabulatable):
            raise TypeError(
                "MonteCarloOnPolicyIterator.__init__(): The environment {} is not an instance of ReinforcementLearning.environments.attributes.Tabulatable".format(
                    environment))
        if not isinstance(environment, env_attributes.Episodic):
            raise TypeError(
                "MonteCarloOnPolicyIterator.__init__(): The environment {} is not an instance of ReinforcementLearning.environments.attributes.Episodic".format(
                    environment))
        if not isinstance(legal_actions_authority, rl_policy.LegalActionsAuthority):
            raise TypeError(
                "MonteCarloOnPolicyIterator.__init__(): The legal_actions_authority {} is not an instance of ReinforcementLearning.algorithms.LegalActionsAuthority".format(
                    legal_actions_authority))
        self.environment = copy.deepcopy(environment)
        self.legal_actions_authority = legal_actions_authority
        self.epsilon = epsilon
        self.gamma = gamma
        self.number_of_iterations = number_of_iterations
        self.episode_maximum_length = episode_maximum_length

    def IteratePolicy(self):
        stateAction_to_value = {}
        stateAction_to_returns = {}
        states_list = list(self.environment.StatesSet())
        for state in states_list:
            legal_actions = list(self.legal_actions_authority.LegalActions(state))
            for legal_action in legal_actions:
                stateAction_to_value[(state, legal_action)] = np.random.normal()
                stateAction_to_returns[(state, legal_action)] = rewards.RunningSum()
        policy = rl_policy.EpsilonGreedy(self.epsilon, stateAction_to_value)

        for iteration in range(self.number_of_iterations):
            episode = self.environment.Episode(
                policy, maximum_number_of_steps=self.episode_maximum_length
            )
            rewards_list = [episode[ndx] for ndx in
                            range(len(episode)) if ndx % 3 == 0]
            rewards_list = rewards_list[1:]  # Get rid of reward 0, as it is not the consequence of an action
            stateAction_pairs = self.environment.StateActionPairs(episode)
            if len(stateAction_pairs) != len(rewards_list):
                raise ValueError("MonteCarloOnPolicyIterator.IteratePolicy(): len(stateAction_pairs) ({}) != len(rewards_list) ({})".format(
                        len(stateAction_pairs),len(rewards_list)))
            stateAction_to_occurred = {stateAction: False for stateAction in stateAction_pairs}
            for stateActionNdx in range(len(stateAction_pairs)):
                (state, action) = stateAction_pairs[stateActionNdx]
                if not stateAction_to_occurred[(state, action)]:
                    return_value = rewards.Return(rewards_list[stateActionNdx:], self.gamma)
                    stateAction_to_returns[(state, action)].Append(return_value)
                    stateAction_to_value[(state, action)] = stateAction_to_returns[(state, action)].Average()
                    stateAction_to_occurred[(state, action)] = True

            policy = rl_policy.EpsilonGreedy(self.epsilon, stateAction_to_value)

        return policy