import numpy
import itertools

class Solver:
    def __init__(self,
                 numberOfObservations,
                 numberOfActions,
                 gamma,
                 learningRate,
                 environment,  # Must implement .Step(action), .Reset()
                 defaultValue, # Fill value for the Q matrix. It should be higher than the penalty given by the environment for a neutral move, in order to favor exploration of unseen cases
                 ):
        self.numberOfObservations = numberOfObservations
        self.numberOfActions = numberOfActions
        self.gamma = gamma
        self.learningRate = learningRate
        self.environment = environment
        self.Q = numpy.zeros([numberOfObservations, numberOfActions])
        self.Q.fill(defaultValue)

    def BestAction(self, observation):
        bestAction = numpy.argmax(self.Q[observation])
        #print ("BestAction(): bestAction = {}".format(bestAction))
        return bestAction

    def UpdateQWithOneEpisode(self):
        done = False
        rewardSum = 0
        observation = self.environment.Reset()
        while not done:
            action = self.BestAction(observation)
            nextObservation, reward, done, info = self.environment.Step(action)
            self.Q[observation, action] += self.learningRate * \
                (reward + self.gamma * numpy.max(self.Q[nextObservation]) - self.Q[observation, action])
            rewardSum += reward
            observation = nextObservation
        return rewardSum

    def Solve(self, numberOfEpisodes, writeToConsole=True):
        for episode in range(numberOfEpisodes):
            rewardSum = self.UpdateQWithOneEpisode()
            if writeToConsole:
                print ("{}. rewardSum = {}".format(episode, rewardSum))
