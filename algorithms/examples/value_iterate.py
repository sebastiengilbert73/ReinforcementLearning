import logging
import argparse
import ReinforcementLearning.algorithms.policy as rl_policy
import ReinforcementLearning.algorithms.value_iteration as value_iteration
import ReinforcementLearning.environments as environments
from ReinforcementLearning.environments import gridworlds
from ReinforcementLearning.environments import jacks_car_rental
from ReinforcementLearning.environments import gamblers_problem
from ReinforcementLearning.environments import frozen_lake
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--randomSeed', help="The seed for the random module. Default: 0", type=int, default=0)
parser.add_argument('--environment', help="The environment to use. Default: 'FrozenLake'", default='FrozenLake')
parser.add_argument('--legalActionsAuthority', help="The authority that filters the legal actions. Default: 'AllActionsLegal'", default='AllActionsLegal')
parser.add_argument('--gamma', help="The discount factor. Default: 0.9", type=float, default=0.9)
parser.add_argument('--minimumChange', help="For the value iterator, the minimum value change to keep iterating. Default: 0.01", type=float, default=0.01)
parser.add_argument('--maximumNumberOfIterations', help="For the value iterator, the maximum number of iterations. Default: 1000", type=int, default=1000)
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s [%(levelname)s] %(message)s')
random.seed(args.randomSeed)


def main():
    logging.info("value_iterate.py main(): environment = {}".format(args.environment))

    # Create the environment
    # It must implement gym.Env, ActionsSet(), StatesSet()
    environment = None
    if args.environment.lower() == 'gridworld1':
        environment = gridworlds.GridWorld1()
    elif args.environment.lower() == 'gridworld2x2':
        environment = gridworlds.GridWorld2x2()
    elif args.environment.lower() == 'jackscarrental':
        environment = jacks_car_rental.JacksCarRental('original')
    elif args.environment.lower() == 'jackscarrental4.4':
        environment = jacks_car_rental.JacksCarRental('exercise_4.4')
    elif args.environment.lower() == 'gamblersproblem':
        environment = gamblers_problem.GamblersProblem(heads_probability=0.4)
    if args.environment.lower() == 'frozenlake':
        environment = frozen_lake.FrozenLake()
    else:
        raise NotImplementedError("main(): Not implemented environment '{}'".format(args.environment))

    # Create the authority to filter the legal actions
    actions_set = environment.ActionsSet()
    legal_actions_authority = None
    if args.legalActionsAuthority.lower() == 'allactionslegal':
        legal_actions_authority = rl_policy.AllActionsLegalAuthority(actions_set)
    elif args.legalActionsAuthority.lower() == 'jackspossiblemoves':
        legal_actions_authority = jacks_car_rental.JacksPossibleMoves()
    elif args.legalActionsAuthority.lower() == 'gamblerspossiblestakes':
        legal_actions_authority = gamblers_problem.GamblersPossibleStakes()
    else:
        raise NotImplementedError(
            "main(): Not implemented legal actions authority '{}'".format(args.legalActionsAuthority))

    # Create the value iterator
    value_iterator = value_iteration.ValueIterator(
        environment=environment,
        legal_actions_authority=legal_actions_authority,
        gamma=args.gamma,
        minimum_change = args.minimumChange,
        maximum_number_of_iterations = args.maximumNumberOfIterations
    )

    # Iterate value
    logging.info("Starting value iteration...")
    policy = value_iterator.Iterate()
    logging.info("Done!")
    logging.info("policy.state_to_most_valuable_action = \n{}".format(policy.state_to_most_valuable_action))

if __name__ == '__main__':
    main()