import numpy
import itertools

class Solver:
    def __init__(self,
                 numberOfStates,
                 numberOfActions,
                 gamma,
                 environment, # Must implement .Set(state): use env.env.s, .Step(action)
                 valueFill,
                 ):
        self.numberOfStates = numberOfStates
        self.numberOfActions = numberOfActions
        self.gamma = gamma
        self.environment = environment
        self.policy = numpy.zeros(numberOfStates, dtype=int)
        self.value = numpy.zeros(numberOfStates, dtype=float)
        self.value.fill(valueFill)


    def BestAction(self, state):
        bestAction = None
        highestValue = float('-inf')

        for action in range(self.numberOfActions):
            self.environment.Set(state)
            newState, reward, done, info = self.environment.Step(action)
            value = reward + self.gamma * self.value[newState]
            if value > highestValue:
                highestValue = value
                bestAction = action
        return bestAction

    def UpdateStates(self):
        biggestValueChange = 0
        numberOfPolicyChanges = 0
        for state in range(self.numberOfStates):
            initialValue = self.value[state]
            initialPolicyAction = self.policy[state]
            action = self.BestAction(state)
            self.environment.Set(state)
            newState, reward, done, info = self.environment.Step(action)
            self.value[state] = reward + self.gamma * self.value[newState]
            self.policy[state] = action
            valueChange = numpy.abs(self.value[state] - initialValue)
            if valueChange > biggestValueChange:
                biggestValueChange = valueChange
            if self.policy[state] != initialPolicyAction:
                numberOfPolicyChanges += 1
        return biggestValueChange, numberOfPolicyChanges

    def Solve(self, valueChangeThreshold, writeToConsole=True):
        for updateNdx in itertools.count():
            biggestValueChange, numberOfPolicyChanges = self.UpdateStates()
            if writeToConsole:
                print ("{}. biggestValueChange = {}; numberOfPolicyChanges = {}".format(updateNdx + 1, biggestValueChange, numberOfPolicyChanges))
            if biggestValueChange < valueChangeThreshold:
                break
        return self.policy, self.value

