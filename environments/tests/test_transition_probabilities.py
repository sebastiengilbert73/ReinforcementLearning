import logging
import argparse
import ReinforcementLearning.algorithms.policy as policy
import ReinforcementLearning.environments as environments
from ReinforcementLearning.environments import gridworlds
from ReinforcementLearning.environments import jacks_car_rental
import random

parser = argparse.ArgumentParser()
parser.add_argument('--randomSeed', help="The seed for the random module. Default: 0", type=int, default=0)
parser.add_argument('--environment', help="The environment to use. Default: 'gridworld1'", default='gridworld1')
parser.add_argument('--legalActionsAuthority', help="The authority that filters the legal actions. Default: 'AllActionsLegal'", default='AllActionsLegal')
parser.add_argument('--state', help="The state the environment must be in. Default: 0", type=int, default=0)
parser.add_argument('--action', help="The action to test. Default: 0", type=int, default=0)
#parser.add_argument('--policy', help="The policy. Default: 'random'", default='random')
#parser.add_argument('--gamma', help="The discount factor. Default: 0.9", type=float, default=0.9)
#parser.add_argument('--minimumChange', help="The minimum value change to keep iterating. Default: 0.01", type=float, default=0.01)
#parser.add_argument('--numberOfSelectionsPerState', help="The number of tried selections per state. Should be 1 if the policy and the environment are deterministic. Default: 100", type=int, default=100)
#parser.add_argument('--maximumNumberOfIterations', help="The maximum number of iterations. Default: 1000", type=int, default=1000)
#parser.add_argument('--initialValue', help="The initial value for all states. Default: 0", type=float, default=0)
#parser.add_argument('--epsilon', help="For epsilon-greedy policies, the probability of choosing a random action. Default: 0.1", type=float, default=0.1)
parser.add_argument('--numberOfTrials', help="The number of trials for the action. For deterministic environments, should be 1. Default: 100", type=int, default=100)

args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s [%(levelname)s] %(message)s')
random.seed(args.randomSeed)


def main():
    logging.info("test_transition_probabilities.py  main()\t\tenvironment = {}\t\tlegalActionsAuthority = {}".format(args.environment, args.legalActionsAuthority))


if __name__ == '__main__':
    main()

    # Create the environment
    # It must implement gym.Env, ActionsSet(), StatesSet(), SetState(s), TransitionProbabilitiesAndRewards(action)
    environment = None
    if args.environment.lower() == 'gridworld1':
        environment = gridworlds.GridWorld1()
    elif args.environment.lower() == 'gridworld2x2':
        environment = gridworlds.GridWorld2x2()
    elif args.environment.lower() == 'jackscarrental':
        environment = jacks_car_rental.JacksCarRental()
    else:
        raise NotImplementedError("main(): Not implemented environment '{}'".format(args.environment))

    # Create the authority to filter the legal actions
    actions_set = environment.ActionsSet()
    legal_actions_authority = None
    if args.legalActionsAuthority.lower() == 'allactionslegal':
        legal_actions_authority = policy.AllActionsLegalAuthority(actions_set)
    elif args.legalActionsAuthority.lower() == 'jackspossiblemoves':
        legal_actions_authority = jacks_car_rental.JacksPossibleMoves()
    else:
        raise NotImplementedError(
            "main(): Not implemented legal actions authority '{}'".format(args.legalActionsAuthority))

    # Get the states set and the actions set
    states_set = environment.StatesSet()
    actions_set = environment.ActionsSet()
    if args.state not in states_set:
        raise ValueError("main(): args.state ({}) is not is the set of states ({})".format(args.state, states_set))

    # Set the environment state
    environment.SetState(args.state)
    legal_actions = legal_actions_authority.LegalActions(args.state)
    if args.action not in legal_actions:
        raise ValueError("main(): args.action ({}) is not in the actions set ({})".format(args.action, actions_set))

    new_state_to_numberOfOccurrences = {s: 0 for s in states_set}
    new_state_to_rewardSum = {s: 0 for s in states_set}
    for trialNdx in range(args.numberOfTrials):
        environment.SetState(args.state)
        new_state, reward, done, info = environment.step(args.action)
        new_state_to_numberOfOccurrences[new_state] += 1
        new_state_to_rewardSum[new_state] += reward
    simulated_new_state_to_probability_reward = {s: (new_state_to_numberOfOccurrences[s]/args.numberOfTrials,
                                                     new_state_to_rewardSum[s]/args.numberOfTrials)
                                                 for s in states_set}

    # Call TransitionProbabilitiesAndRewards(action)
    environment.SetState(args.state)
    coded_new_state_to_probability_reward = environment.TransitionProbabilitiesAndRewards(args.action)

    # Display the comparison
    #print ("new_state\t\tsimulated probability\tcoded_probability\t\tsimulated expected reward\tcoded expected reward")
    print("{:<30}{:<30}{:<30}{:<30}{:<30}".format("new_state", "simulated probability", "coded_probability", "simulated expected reward", "coded expected reward"))
    for state in states_set:
        print("     {:<30}{:<30}{:<30}{:<30}{:<30}".format(state, simulated_new_state_to_probability_reward[state][0],
                                              coded_new_state_to_probability_reward[state][0],
                                              simulated_new_state_to_probability_reward[state][1],
                                              coded_new_state_to_probability_reward[state][1]))

    simulated_probabilities_sum = sum(simulated_new_state_to_probability_reward[s][0] for s in states_set)
    coded_probabilities_sum = sum(coded_new_state_to_probability_reward[s][0] for s in states_set)
    print ("\n{:>50}{:>30}".format(simulated_probabilities_sum, coded_probabilities_sum))