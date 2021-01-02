import gym
import abc

class DynamicProgrammingEnv(abc.ABC, gym.Env):
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
    def StatesSet(self):
        pass

    @abc.abstractmethod
    def SetState(self, state):
        pass

    @abc.abstractmethod
    def ActionsSet(self):
        pass

    @abc.abstractmethod
    def TransitionProbabilitiesAndRewards(self, action):
        pass  # return newState_to_probabilityAndReward_dict