import logging
import argparse
import ReinforcementLearning.algorithms.policy as policy
import ReinforcementLearning.environments as environments
from ReinforcementLearning.environments import gridworlds
from ReinforcementLearning.environments import jacks_car_rental
from ReinforcementLearning.environments import gamblers_problem
import random
import ast

parser = argparse.ArgumentParser()
parser.add_argument('--randomSeed', help="The seed for the random module. Default: 0", type=int, default=0)
parser.add_argument('--environment', help="The environment to use. Default: 'gridworld1'", default='gridworld1')
parser.add_argument('--legalActionsAuthority', help="The authority that filters the legal actions. Default: 'AllActionsLegal'", default='AllActionsLegal')
parser.add_argument('--state', help="The state the environment must be in. -1 to scan and compare. Default: 0", type=int, default=0)
parser.add_argument('--action', help="The action to test. Default: 0", type=int, default=0)
parser.add_argument('--numberOfTrials', help="The number of trials for the action. For deterministic environments, should be 1. Default: 1000", type=int, default=1000)
parser.add_argument('--newStatesList', help="The list of states to display. Default: 'all'", default='all')
parser.add_argument('--maximumProbabilityDelta', help="For scan and compare, the maximum probability difference to highlight. Default: 0.05", type=float, default=0.05)
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s [%(levelname)s] %(message)s')
random.seed(args.randomSeed)
if args.newStatesList == 'all':
    newStatesList = None
else:
    newStatesList = ast.literal_eval(args.newStatesList)

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
    elif args.environment.lower() == 'gamblersproblem':
        environment = gamblers_problem.GamblersProblem(heads_probability=0.4)
    else:
        raise NotImplementedError("main(): Not implemented environment '{}'".format(args.environment))

    # Create the authority to filter the legal actions
    actions_set = environment.ActionsSet()
    legal_actions_authority = None
    if args.legalActionsAuthority.lower() == 'allactionslegal':
        legal_actions_authority = policy.AllActionsLegalAuthority(actions_set)
    elif args.legalActionsAuthority.lower() == 'jackspossiblemoves':
        legal_actions_authority = jacks_car_rental.JacksPossibleMoves()
    elif args.legalActionsAuthority.lower() == 'gamblerspossiblestakes':
        legal_actions_authority = gamblers_problem.GamblersPossibleStakes()
    else:
        raise NotImplementedError(
            "main(): Not implemented legal actions authority '{}'".format(args.legalActionsAuthority))

    # Get the states set and the actions set
    states_set = environment.StatesSet()
    actions_set = environment.ActionsSet()
    if args.state != -1 and args.state not in states_set:
        raise ValueError("main(): args.state ({}) is not is the set of states ({})".format(args.state, states_set))
    if newStatesList is None:
        newStatesList = list(states_set)

    # Set the environment state
    if args.state != -1:
        environment.SetState(args.state)
        legal_actions = legal_actions_authority.LegalActions(args.state)
        if args.action not in legal_actions:
            raise ValueError("main(): args.action ({}) is not in the legal actions set ({})".format(args.action, legal_actions))

        new_state_to_numberOfOccurrences = {s: 0 for s in states_set}
        new_state_to_rewardSum = {s: 0 for s in states_set}
        for trialNdx in range(args.numberOfTrials):
            environment.SetState(args.state)
            new_state, reward, done, info = environment.step(args.action)
            new_state_to_numberOfOccurrences[new_state] += 1
            new_state_to_rewardSum[new_state] += reward
        simulated_new_state_to_probability_reward = {}
        for s in states_set:
            probability = new_state_to_numberOfOccurrences[s]/args.numberOfTrials
            expected_reward = 0
            if new_state_to_numberOfOccurrences[s] > 0:
                expected_reward = new_state_to_rewardSum[s]/new_state_to_numberOfOccurrences[s]
            simulated_new_state_to_probability_reward[s] = (probability, expected_reward)

        # Call TransitionProbabilitiesAndRewards(action)
        environment.SetState(args.state)
        coded_new_state_to_probability_reward = environment.TransitionProbabilitiesAndRewards(args.action)

        # Display the comparison
        print("{:<30}{:<30}{:<30}{:<30}{:<30}".format("new_state", "simulated probability", "coded_probability", "simulated expected reward", "coded expected reward"))
        simulated_probabilities_sum = 0
        coded_probabilities_sum = 0
        for state in newStatesList:
            simulated_probability = simulated_new_state_to_probability_reward.get(state, (0, 0))[0]
            simulated_reward = simulated_new_state_to_probability_reward.get(state, (0, 0))[1]
            coded_probability = coded_new_state_to_probability_reward.get(state, (0, 0))[0]
            coded_reward = coded_new_state_to_probability_reward.get(state, (0, 0))[1]
            print("     {:<30}{:<30}{:<30}{:<30}{:<30}".format(state, simulated_probability,
                                                  coded_probability,
                                                  simulated_reward,
                                                  coded_reward))
            simulated_probabilities_sum += simulated_probability
            coded_probabilities_sum += coded_probability

        print ("\n{:>50}{:>30}".format(simulated_probabilities_sum, coded_probabilities_sum))

    else:  # Scan and compare
        logging.info("Scan and compare. maximumProbabilityDelta = {}".format(args.maximumProbabilityDelta))
        for origin_state in states_set:
            logging.info("origin_state = {}".format(origin_state))
            legal_actions = legal_actions_authority.LegalActions(origin_state)
            for action in legal_actions:
                new_state_to_numberOfOccurrences = {s: 0 for s in states_set}
                new_state_to_rewardSum = {s: 0 for s in states_set}
                for trialNdx in range(args.numberOfTrials):
                    environment.SetState(origin_state)
                    new_state, reward, done, info = environment.step(action)
                    new_state_to_numberOfOccurrences[new_state] += 1
                    new_state_to_rewardSum[new_state] += reward
                simulated_new_state_to_probability_reward = {}
                for s in states_set:
                    probability = new_state_to_numberOfOccurrences[s] / args.numberOfTrials
                    expected_reward = 0
                    if new_state_to_numberOfOccurrences[s] > 0:
                        expected_reward = new_state_to_rewardSum[s] / new_state_to_numberOfOccurrences[s]
                    simulated_new_state_to_probability_reward[s] = (probability, expected_reward)

                # Call TransitionProbabilitiesAndRewards(action)
                environment.SetState(origin_state)
                coded_new_state_to_probability_reward = environment.TransitionProbabilitiesAndRewards(action)

                # Compare
                for new_state in states_set:
                    simulated_probability = simulated_new_state_to_probability_reward[new_state][0]
                    coded_probability = coded_new_state_to_probability_reward[new_state][0]
                    if abs(simulated_probability - coded_probability) > args.maximumProbabilityDelta:
                        print ("origin_state = {}; action = {}; new_state = {}; simulated_probability = {}; coded_probability = {}".format(origin_state, action, new_state, simulated_probability, coded_probability))