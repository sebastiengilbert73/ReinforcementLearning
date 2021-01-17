import random
import copy
import statistics
import ReinforcementLearning.environments.attributes as env_attributes

class FirstVisitPolicyEvaluator:
    """
    Implements algorithm in 'Reinforcement Learning', Sutton and Barto, p.113
    Environment interface:
        StatesSet()
        reset() -> observation, reward, done, info
        state
        step(action) -> observation, reward, done, info
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
            observationReward_list = self.Episode(policy)
            observation_first_visit_is_encountered = {o: False for (o, r) in observationReward_list}
            for observationNdx in range(len(observationReward_list)):
                (observation, reward) = observationReward_list[observationNdx]
                if observation in states_set and not observation_first_visit_is_encountered[observation]:
                    observationReturn = self.Return(observationReward_list[observationNdx:])
                    state_to_returns_dict[observation].append(observationReturn)
                    observation_first_visit_is_encountered[observation] = True
                    state_to_value_dict[observation] = statistics.mean(state_to_returns_dict[observation])
            if print_iteration and iteration % 100 == 1:
                print('.', end='', flush=True)
        if print_iteration:
            print()
        return state_to_value_dict

    def Episode(self, policy):
        observationReward_list = []
        observation, reward, episode_is_done, _ = self.environment.reset()
        observationReward_list.append((observation, reward))
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