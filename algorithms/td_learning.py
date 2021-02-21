import random
import copy
import statistics
import ReinforcementLearning.environments.attributes as env_attributes
import ReinforcementLearning.algorithms.policy as rl_policy
import ReinforcementLearning.utilities.rewards as rewards
import numpy as np

class TD0PolicyEvaluator:
    def __init__(self,
                 environment,
                 gamma=0.9,
                 alpha=0.05,
                 number_of_episodes=1000,
                 episode_maximum_length=1000
                 ):
        """Implements 'Tabular TD(0) for estimating V^pi', from 'Reinforcement Learning', Sutton and Barto, p. 135"""
        if not isinstance(environment, env_attributes.Tabulatable):
            raise TypeError("TD0PolicyEvaluator.__init__(): The environment type ({}) is not an instance of ReinforcementLearning.environments.attributes.Tabulatable".format(type(environment)))
        if not isinstance(environment, env_attributes.Episodic):
            raise TypeError(
                "TD0PolicyEvaluator.__init__(): The environment type ({}) is not an instance of ReinforcementLearning.environments.attributes.Episodic".format(
                    type(environment)))
        self.environment = copy.deepcopy(environment)
        self.alpha = alpha
        self.gamma = gamma
        self.number_of_episodes = number_of_episodes
        self.episode_maximum_length = episode_maximum_length

    def Evaluate(self, policy):
        states_set = self.environment.StatesSet()
        state_to_value = {}
        for state in states_set:
            state_to_value[state] = np.random.normal()

        for iterationNdx in range(self.number_of_episodes):
            episode = self.environment.Episode(policy)
            print ("episode: {}".format(episode))
            # episode = [reward0, obs0, action1, reward1, obs1, action2, reward2, obs2, ..., obsN-1]
            rewards_list = [episode[ndx] for ndx in range(1, len(episode)) if ndx % 3 == 0]  # Ignore reward0
            stateAction_pairs = self.environment.StateActionPairs(episode)
            for actionNdx in range(len(stateAction_pairs)):
                state, action = stateAction_pairs[actionNdx]
                reward = rewards_list[actionNdx]
                next_state = episode[1 + 3 * (actionNdx + 1) ]
                print ("state = {}; action = {}; next_state = {}".format(state, action, next_state))
                current_state_value = state_to_value[state]  # V(s)
                next_state_value = state_to_value[next_state]  # V(s')
                current_state_new_value = current_state_value + self.alpha * \
                    (reward + self.gamma * next_state_value - current_state_value)
                state_to_value[state] = current_state_new_value
        return state_to_value