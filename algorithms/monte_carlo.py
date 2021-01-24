import random
import copy
import statistics
import ReinforcementLearning.environments.attributes as env_attributes


def Return(reward_list, gamma):
    discounted_sum = 0
    for rewardNdx in range(len(reward_list)):
        discount = pow(gamma, rewardNdx)
        discounted_sum += discount * reward_list[rewardNdx]
    return discounted_sum

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
                    observationReturn = Return(rewards_list[observationNdx:], self.gamma)
                    state_to_returns_dict[observation].append(observationReturn)
                    observation_first_visit_is_encountered[observation] = True
                    state_to_value_dict[observation] = statistics.mean(state_to_returns_dict[observation])
            if print_iteration and iteration % 100 == 1:
                print('.', end='', flush=True)
        if print_iteration:
            print()
        return state_to_value_dict
