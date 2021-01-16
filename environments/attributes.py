import abc
import gym


class GymCompatible(abc.ABC, gym.Env):
    def __init__(self):
        super(gym.Env).__init__()

    @abc.abstractmethod
    def step(self, action):
        pass  # return (observation, reward, done, info_dict)

    @abc.abstractmethod
    def reset(self):
        pass  # return self.state

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
    def SetState(self, state):
        pass

class DynamicProgramming(abc.ABC):
    @abc.abstractmethod
    def TransitionProbabilitiesAndRewards(self, state, action):
        pass

class Tabulatable(abc.ABC):
    @abc.abstractmethod
    def StatesSet(self):
        pass

class TabulatableDP(Tabulatable, DynamicProgramming):
    def __init__(self):
        self.originStateAction_to_newStateToProbabilityReward = {}  # The cached transition probabilities and rewards

    def TransitionProbabilitiesAndRewards(self, state, action):
        if (state, action) not in self.originStateAction_to_newStateToProbabilityReward:
            self.originStateAction_to_newStateToProbabilityReward[(state, action)] = self.ComputeTransitionProbabilitiesAndRewards(state, action)
        return self.originStateAction_to_newStateToProbabilityReward[(state, action)]

    @abc.abstractmethod
    def ComputeTransitionProbabilitiesAndRewards(self, state, action):
        pass  # return newState_to_probabilityAndReward_dict

