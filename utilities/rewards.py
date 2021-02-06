import numpy as np

class RunningSum():
    def __init__(self):
        self.sum = 0
        self.count = 0

    def Append(self, value):
        self.sum += value
        self.count += 1

    def Average(self):
        if self.count < 1:
            raise ValueError("RunningSum.Average(): Attempt to call average when the count is 0")
        return self.sum/self.count


def Return(reward_list, gamma):
    discounted_sum = 0
    for rewardNdx in range(len(reward_list)):
        discount = pow(gamma, rewardNdx)
        discounted_sum += discount * reward_list[rewardNdx]
    return discounted_sum