import gym
import abc

class DynamicProgrammingEnv(abc.ABC, gym.Env):
    def __init__(self):
        super(gym.Env).__init__()
        self.state = None
        self.originStateAction_to_newStateToProbabilityReward = {}  # The cached transition probabilities and rewards

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
    def StatesSet(self):
        pass

    @abc.abstractmethod
    def SetState(self, state):
        pass

    @abc.abstractmethod
    def ActionsSet(self):
        pass

    def TransitionProbabilitiesAndRewards(self, action):
        if (self.state, action) not in self.originStateAction_to_newStateToProbabilityReward:
            self.originStateAction_to_newStateToProbabilityReward[(self.state, action)] = self.ComputeTransitionProbabilitiesAndRewards(action)
        return self.originStateAction_to_newStateToProbabilityReward[(self.state, action)]

    @abc.abstractmethod
    def ComputeTransitionProbabilitiesAndRewards(self, action):
        pass  # return newState_to_probabilityAndReward_dict