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
    def Episode(self, policy, start_state=None, maximum_number_of_steps=None):
        observationReward_list = []
        observation = None
        reward = 0
        episode_is_done = False
        info = {}
        if start_state is not None:
            if not isinstance(self, ExplorationStarts):
                raise TypeError("Episodic.Episode(): The start state is not None and the environment is not an instance of ReinforcementLearning.environments.attributes.ExplorationStarts")
            (observation, reward, episode_is_done, info) = self.SetState(start_state)
            observationReward_list.append((observation, reward))
        else:
            (observation, reward, episode_is_done, info) = self.reset()
            observationReward_list.append((observation, reward))
        if maximum_number_of_steps is None:
            maximum_number_of_steps = sys.maxsize

        while not episode_is_done and len(observationReward_list) < maximum_number_of_steps:
            action = policy.Select(observation)
            observation, reward, episode_is_done, info = self.step(action)
            observationReward_list.append((observation, reward))
        return observationReward_list