import logging
import argparse
import ReinforcementLearning.algorithms.policy as rl_policy
import ReinforcementLearning.algorithms.monte_carlo as monte_carlo
import ReinforcementLearning.environments as environments
from ReinforcementLearning.environments import blackjack
import random
import os

parser = argparse.ArgumentParser()
parser.add_argument('--randomSeed', help="The seed for the random module. Default: 0", type=int, default=0)
parser.add_argument('--environment', help="The environment to use. Default: 'Blackjack-v0'", default='Blackjack-v0')
parser.add_argument('--legalActionsAuthority', help="The authority that filters the legal actions. Default: 'AllActionsLegal'", default='AllActionsLegal')
parser.add_argument('--epsilon', help='The probability of selecting randomly, for the epsilon-greedy policy. Default: 0.1', type=float, default=0.1)
parser.add_argument('--gamma', help="The discount factor. Default: 1.0", type=float, default=1.0)
parser.add_argument('--numberOfIterations', help="The number of iterations. Default: 100", type=int, default=100)
parser.add_argument('--episodeMaximumLength', help="The episode maximum length. Default: 100", type=int, default=100)
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s [%(levelname)s] %(message)s')
random.seed(args.randomSeed)

def main():
    logging.info("monte_carlo_on_policy_iteration.py main()\tenvironment = {}".format(args.environment))

    # Create the environment
    environment = None
    if args.environment.lower() == 'blackjack-v0':
        environment = blackjack.Blackjack()
    elif args.environment.lower() == 'blackjackes':
        environment = blackjack.BlackjackES()
    else:
        raise NotImplementedError("main(): Not implemented environment '{}'".format(args.environment))

    # Create the authority to filter the legal actions
    actions_set = environment.ActionsSet()
    legal_actions_authority = None
    if args.legalActionsAuthority.lower() == 'allactionslegal':
        legal_actions_authority = rl_policy.AllActionsLegalAuthority(actions_set)
    else:
        raise NotImplementedError("main(): Not implemented legal actions authority '{}'".format(args.legalActionsAuthority))

    # Create the Monte-Carlo on policy iterator
    policy_iterator = monte_carlo.MonteCarloOnPolicyIterator(
        environment=environment,
        legal_actions_authority=legal_actions_authority,
        epsilon=args.epsilon,
        gamma=args.gamma,
        number_of_iterations=args.numberOfIterations,
        episode_maximum_length=args.episodeMaximumLength
    )
    logging.info("Starting iteration...")
    policy = policy_iterator.IteratePolicy()
    logging.info("Done!")
    print ("policy = \n{}".format(policy))

    WriteOutput(args.environment, policy, environment)

def WriteOutput(environment_name, policy, environment):
    if environment_name.lower() == 'blackjackes':
        output_filepath = os.path.join("/tmp/", 'monteCarloOnPolicyIteration_BlackjackES.csv')
        with open(output_filepath, 'w+') as output_file:
            for has_usable_ace in [True, False]:
                output_file.write("dealer_card:,1,2,3,4,5,6,7,8,9,10\n")
                for player_sum in range(21, 11, -1):  # With a usable ace, the player sum is at least 12
                    output_file.write('{}'.format(player_sum))
                    for dealer_card in range(1, 11):
                        state = (player_sum, has_usable_ace, dealer_card)
                        action = policy.Select(state)
                        output_file.write(',{}'.format(action))
                    output_file.write('\n')
                output_file.write('\n')
    elif environment_name.lower() == 'blackjack-v0':
        output_filepath = os.path.join("/tmp/", 'monteCarloOnPolicyIteration_Blackjack-v0.csv')
        with open(output_filepath, 'w+') as output_file:
            for has_usable_ace in [True, False]:
                output_file.write("dealer_card:,1,2,3,4,5,6,7,8,9,10\n")
                for player_sum in range(21, 11, -1):  # With a usable ace, the player sum is at least 12
                    output_file.write('{}'.format(player_sum))
                    for dealer_card in range(1, 11):
                        state = (player_sum, has_usable_ace, dealer_card)
                        action = policy.Select(state)
                        output_file.write(',{}'.format(action))
                    output_file.write('\n')
                output_file.write('\n')
    else:
        logging.info("We're not writing anything for environment {}".format(environment_name))

if __name__ == '__main__':
    main()