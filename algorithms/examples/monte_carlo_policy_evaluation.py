import logging
import argparse
import ReinforcementLearning.algorithms.policy as rl_policy
import ReinforcementLearning.algorithms.monte_carlo as monte_carlo
#import ReinforcementLearning.algorithms.dp_iteration as dp_iteration
import ReinforcementLearning.environments as environments
#from ReinforcementLearning.environments import gridworlds
#from ReinforcementLearning.environments import jacks_car_rental
#from ReinforcementLearning.environments import gamblers_problem
from ReinforcementLearning.environments import blackjack
import random

parser = argparse.ArgumentParser()
parser.add_argument('--randomSeed', help="The seed for the random module. Default: 0", type=int, default=0)
parser.add_argument('--environment', help="The environment to use. Default: 'blackjack'", default='blackjack')
parser.add_argument('--legalActionsAuthority', help="The authority that filters the legal actions. Default: 'AllActionsLegal'", default='AllActionsLegal')
parser.add_argument('--policy', help="The policy. Default: 'random'", default='random')
parser.add_argument('--gamma', help="The discount factor. Default: 0.9", type=float, default=0.9)
parser.add_argument('--minimumChange', help="The minimum value change to keep iterating. Default: 0.01", type=float, default=0.01)
parser.add_argument('--numberOfIterations', help="The number of iterations. Default: 100", type=int, default=100)
parser.add_argument('--initialValue', help="The initial value for all states. Default: 0", type=float, default=0)
parser.add_argument('--episodeMaximumLength', help="The episode maximum length. Default: 100", type=int, default=100)
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s [%(levelname)s] %(message)s')
random.seed(args.randomSeed)

def main():
    logging.info("monte_carlo_policy_evaluation.py main()\tenvironment = {}".format(args.environment))

    # Create the environment
    environment = None
    if args.environment.lower() == 'blackjack':
        environment = blackjack.Blackjack()
    else:
        raise NotImplementedError("main(): Not implemented environment '{}'".format(args.environment))

    # Create the authority to filter the legal actions
    actions_set = environment.ActionsSet()
    legal_actions_authority = None
    if args.legalActionsAuthority.lower() == 'allactionslegal':
        legal_actions_authority = rl_policy.AllActionsLegalAuthority(actions_set)
    else:
        raise NotImplementedError("main(): Not implemented legal actions authority '{}'".format(args.legalActionsAuthority))

    # Create the evaluated policy
    states_set = environment.StatesSet()
    evaluated_policy = None
    if args.policy.lower() == 'random':
        evaluated_policy = rl_policy.Random(legal_actions_authority)
    elif args.policy.lower() == 'hitupto19':
        evaluated_policy = blackjack.HitUpTo(19)
    else:
        raise NotImplementedError(
            "main(): Not implemented policy '{}'".format(args.policy))

    # Create the Monte-Carlo policy evaluator
    policy_evaluator = monte_carlo.FirstVisitPolicyEvaluator(
        environment=environment,
        gamma=args.gamma,
        number_of_iterations=args.numberOfIterations,
        initial_value=args.initialValue,
        episode_maximum_length=args.episodeMaximumLength
    )
    logging.info("Starting evaluation...")
    state_to_value_dict = policy_evaluator.Evaluate(evaluated_policy, print_iteration=True)
    logging.info("Done!")
    print ("state_to_value_dict = \n{}".format(state_to_value_dict))


if __name__ == '__main__':
    main()