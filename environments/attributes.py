import abc
import gym
import sys


class GymCompatible(abc.ABC, gym.Env):
    def __init__(self):
        super(gym.Env).__init__()

    @abc.abstractmethod
    def step(self, action):
        pass  # return (observation, reward, done, info_dict)

    @abc.abstractmethod
    def reset(self):
        pass  # return (observation, reward, done, info_dict)

    @abc.abstractmethod
    def render(self, mode):
        pass

    @abc.abstractmethod
    def close(self):
        pass

    @abc.abstractmethod
    def seed(self, seed):
        pass

    @abc.abstractmethod
    def ActionsSet(self):
        pass

class ExplorationStarts(abc.ABC):
    @abc.abstractmethod
    def SetState(self, state):  # return (observation, reward, done, info_dict)
        pass

class TransitionDynamics(abc.ABC):
    @abc.abstractmethod
    def TransitionProbabilitiesAndRewards(self, state, action):
        pass

class Tabulatable(abc.ABC):
    @abc.abstractmethod
    def StatesSet(self):
        pass

class DynamicProgramming(Tabulatable, TransitionDynamics):
    def __init__(self):
        self.originStateAction_to_newStateToProbabilityReward = {}  # The cached transition probabilities and rewards

    def TransitionProbabilitiesAndRewards(self, state, action):
        if (state, action) not in self.originStateAction_to_newStateToProbabilityReward:
            self.originStateAction_to_newStateToProbabilityReward[(state, action)] = self.ComputeTransitionProbabilitiesAndRewards(state, action)
        return self.originStateAction_to_newStateToProbabilityReward[(state, action)]

    @abc.abstractmethod
    def ComputeTransitionProbabilitiesAndRewards(self, state, action):
        pass  # return newState_to_probabilityAndReward_dict

class Episodic(GymCompatible):
    def Episode(self, policy,
                start_state=None,
                start_action=None,
                maximum_number_of_steps=None):
        # returns [reward0, obs0, action1, reward1, obs1, action2, reward2, obs2, ..., obsN-1]
        observationActionReward_list = []
        observation = None
        reward = 0
        episode_is_done = False
        info = {}
        if start_state is not None:
            if not isinstance(self, ExplorationStarts):
                raise TypeError("Episodic.Episode(): The start state is not None and the environment is not an instance of ReinforcementLearning.environments.attributes.ExplorationStarts")
            (observation, reward, episode_is_done, info) = self.SetState(start_state)
            observationActionReward_list.append(reward)
            observationActionReward_list.append(observation)
        else:
            (observation, reward, episode_is_done, info) = self.reset()
            observationActionReward_list.append(reward)
            observationActionReward_list.append(observation)
        number_of_steps = 0
        if maximum_number_of_steps is None:
            maximum_number_of_steps = sys.maxsize
        if start_action is not None:
            observationActionReward_list.append(start_action)
            observation, reward, episode_is_done, info = self.step(start_action)
            observationActionReward_list.append(reward)
            observationActionReward_list.append(observation)
            number_of_steps += 1
        while not episode_is_done and number_of_steps < maximum_number_of_steps:
            action = policy.Select(observation)
            observationActionReward_list.append(action)
            observation, reward, episode_is_done, info = self.step(action)
            observationActionReward_list.append(reward)
            observationActionReward_list.append(observation)
            number_of_steps += 1
        return observationActionReward_list

    @staticmethod
    def StateActionPairs(episode):
        # episode = [reward0, obs0, action1, reward1, obs1, action2, reward2, obs2, ..., obsN - 1]
        if len(episode)%3 != 2:
            raise ValueError("Episodic.StateActionPairs(): The length of the episode ({}) modulo 3 is not 2".format(len(episode)))
        stateAction_pairs = []
        number_of_actions = len(episode)//3
        for stepNdx in range(number_of_actions):
            stateNdx = 3 * stepNdx + 1
            actionNdx = 3 * stepNdx + 2
            stateAction_pairs.append((episode[stateNdx], episode[actionNdx]))
        return stateAction_pairs