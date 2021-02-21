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
    Gets rid of the dependence on exploration starts
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

class MonteCarloOffPolicyIterator():
    """
    Implements 'An off-policy Monte Carlo control algorithm', 'Reinforcement Learning', Sutton & Barto p. 127
    Gets rid of the dependence on exploration starts.
    It iterates a deterministic policy based on episodes generated by an epsilon-soft policy pi'
    Environment interface:
        Tabulatable
        Episodic
    """
    def __init__(self,
                 environment,
                 legal_actions_authority,
                 gamma=0.9,
                 number_of_iterations=1000,
                 episode_maximum_length=1000
                 ):
        if not isinstance(environment, env_attributes.Tabulatable):
            raise TypeError(
                "MonteCarloOffPolicyIterator.__init__(): The environment {} is not an instance of ReinforcementLearning.environments.attributes.Tabulatable".format(
                    environment))
        if not isinstance(environment, env_attributes.Episodic):
            raise TypeError(
                "MonteCarloOffPolicyIterator.__init__(): The environment {} is not an instance of ReinforcementLearning.environments.attributes.Episodic".format(
                    environment))
        if not isinstance(legal_actions_authority, rl_policy.LegalActionsAuthority):
            raise TypeError(
                "MonteCarloOffPolicyIterator.__init__(): The legal_actions_authority {} is not an instance of ReinforcementLearning.algorithms.LegalActionsAuthority".format(
                    legal_actions_authority))
        self.environment = copy.deepcopy(environment)
        self.legal_actions_authority = legal_actions_authority
        self.gamma = gamma
        self.number_of_iterations = number_of_iterations
        self.episode_maximum_length = episode_maximum_length

    def IteratePolicy(self, epsilon_soft_policy):
        stateAction_to_value = {}
        stateAction_to_numerator = {}
        stateAction_to_denominator = {}
        state_to_mostValuableAction = {}
        states_list = list(self.environment.StatesSet())
        for state in states_list:
            legal_actions = list(self.legal_actions_authority.LegalActions(state))
            for legal_action in legal_actions:
                stateAction_to_value[(state, legal_action)] = np.random.normal()
                stateAction_to_numerator[(state, legal_action)] = 0
                stateAction_to_denominator[(state, legal_action)] = 0
            state_to_mostValuableAction[state] = legal_actions[0]  # Arbitrarily select the first action
        policy = rl_policy.Greedy(state_to_mostValuableAction, self.legal_actions_authority)

        for iterationNdx in range(self.number_of_iterations):
            episode = self.environment.Episode(epsilon_soft_policy, maximum_number_of_steps=self.episode_maximum_length)
            # episode = [reward0, obs0, action1, reward1, obs1, action2, reward2, obs2, ..., obsN-1]
            rewards_list = [episode[ndx] for ndx in range(1, len(episode)) if ndx % 3 == 0]  # Ignore reward0
            stateAction_pairs = self.environment.StateActionPairs(episode)
            # tau: the latest time at which a_tau != pi(s_tau)
            tau = None
            for candidateNdx in reversed(range(len(stateAction_pairs))):
                state = stateAction_pairs[candidateNdx][0]
                epsilon_soft_action = stateAction_pairs[candidateNdx][1]
                policy_action = policy.Select(state)  # policy is deterministic
                if epsilon_soft_action != policy_action:
                    tau = candidateNdx
                    break  # Stop searching
            if tau is not None:  # tau would be None if all actions taken by epsilon_soft_policy were identical to actions taken by policy
                stateAction_to_occurred = {stateAction: False for stateAction in stateAction_pairs[tau:]}
                for time in range(tau, len(stateAction_pairs)):
                    (state, action) = stateAction_pairs[time]
                    if not stateAction_to_occurred[(state, action)]:  # First occurrence
                        weight = 1
                        for k in range(time + 1, len(stateAction_pairs)):
                            epsilon_soft_prob = epsilon_soft_policy.ActionProbabilities(state)[action]
                            weight = weight * 1.0/epsilon_soft_prob
                        return_t = rewards.Return(rewards_list[time:], self.gamma)
                        stateAction_to_numerator[(state, action)] = stateAction_to_numerator[(state, action)] + weight * return_t
                        stateAction_to_denominator[(state, action)] = stateAction_to_denominator[(state, action)] + weight
                        stateAction_to_value[(state, action)] = stateAction_to_numerator[(state, action)]/stateAction_to_denominator[(state, action)]
                # Update the deterministic policy
                for state in states_list:
                    legal_actions = list(self.legal_actions_authority.LegalActions(state))
                    most_valuable_actions = None
                    highest_value = float('-inf')
                    for legal_action in legal_actions:
                        if stateAction_to_value[(state, legal_action)] > highest_value:
                            highest_value = stateAction_to_value[(state, legal_action)]
                            most_valuable_actions = [legal_action]
                        elif stateAction_to_value[(state, legal_action)] == highest_value:
                            most_valuable_actions.append(legal_action)
                    state_to_mostValuableAction[state] = random.choice(most_valuable_actions)
                policy = rl_policy.Greedy(state_to_mostValuableAction, self.legal_actions_authority)
        return policy