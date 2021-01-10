import random
import copy
import statistics

class FirstVisitPolicyEvaluator:
    """
    Implements algorithm in 'Reinforcement Learning', Sutton and Barto, p.113
    """
    def __init__(self, environment,
                 gamma=0.9,
                 number_of_iterations=1000,
                 initial_value=0,
                 episode_maximum_length=1000):
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

        for iteration in self.number_of_iterations:
            # Generate an episode
            observationReward_list = self.Episode(policy)
            observation_first_visit_is_encountered = {o: False for (o, r) in observationReward_list}
            for observationNdx in range(len(observationReward_list)):
                (observation, reward) = observationReward_list[observationNdx]
                if not observation_first_visit_is_encountered[observation]:
                    observationReturn = self.Return(observationReward_list[observationNdx:])
                    state_to_returns_dict[observation].append(observationReturn)
                    observation_first_visit_is_encountered[observation] = True
                    state_to_value_dict[observation] = statistics.mean(state_to_returns_dict[observation])
            if print_iteration:
                print('.', end='', flush=True)
        if print_iteration:
            print()
        return state_to_value_dict

    def Episode(self, policy):
        observationReward_list = []
        episode_is_done = False
        self.environment.reset()
        while not episode_is_done and len(observationReward_list) < self.episode_maximum_length:
            action = policy.Select(self.environment.state)
            observation, reward, episode_is_done, _ = self.environment.step(action)
            observationReward_list.append((observation, reward))
        return observationReward_list

    def Return(self, observationReward_list):
        discounted_sum = 0
        for observationNdx in range(len(observationReward_list)):
            discount = pow(self.gamma, observationNdx)
            discounted_sum += discount * observationReward_list[observationNdx][1]
        return discounted_sum