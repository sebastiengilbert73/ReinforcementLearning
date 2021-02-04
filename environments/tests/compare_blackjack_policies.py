import logging
import argparse
import ReinforcementLearning.algorithms.policy as rl_policy
import ReinforcementLearning.algorithms.monte_carlo as monte_carlo
import ReinforcementLearning.environments as environments
from ReinforcementLearning.environments import blackjack
import random
import os
import ast

parser = argparse.ArgumentParser()
parser.add_argument('--randomSeed', help="The seed for the random module. Default: 0", type=int, default=0)
parser.add_argument('--environment', help="The Blackjack environment. Default: BlackjackES", default='BlackjackES')
parser.add_argument('--numberOfEpisodes', help="The number of episodes to run. Default: 1000", type=int, default=1000)
parser.add_argument('--startState', help="The start state (player_sum, has_usable_ace, dealer_card). Default: 'None'", default='None')
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s [%(levelname)s] %(message)s')
random.seed(args.randomSeed)


def main():
    logging.info("compare_blackjack_policies.py main()")
    environment = None
    if args.environment.lower() == 'blackjackes':
        environment = blackjack.BlackjackES()
    elif args.environment.lower() == 'blackjack-v0':
        environment = blackjack.Blackjack()
    sutton_barto_policy = blackjack.BJSuttonBarto()
    iterated_policy = blackjack.Iterated()
    states_list = list(environment.StatesSet())

    start_state = None
    if args.startState.lower() == 'none':
        start_state = None
    else:
        start_state = ast.literal_eval(args.startState)
        if not isinstance(start_state, tuple):
            raise TypeError("Start state ({}) is not a tuple".format(start_state))
        if len(start_state) != 3:
            raise ValueError("The length of the start state '{}' is not 3".format(start_state))

    episode1 = environment.Episode(sutton_barto_policy, start_state=start_state)
    print("Episode with sutton_barto_policy: {}".format(episode1))

    episode2 = environment.Episode(iterated_policy, start_state=start_state)
    print("Episode with iterated_policy: {}".format(episode2))

    sutton_barto_sum = 0
    iterated_sum = 0
    for episodeNdx in range(args.numberOfEpisodes):
        if args.startState.lower() == 'none':
            #start_state = random.choice(states_list)
            start_state = None
        #print ("start_state = {}".format(start_state))
        sutton_barto_episode = environment.Episode(sutton_barto_policy, start_state=start_state)
        iterated_episode = environment.Episode(iterated_policy, start_state=start_state)

        sutton_barto_reward = sutton_barto_episode[-2]
        sutton_barto_sum += sutton_barto_reward
        iterated_reward = iterated_episode[-2]
        iterated_sum += iterated_reward

    sutton_barto_expected_value = sutton_barto_sum/args.numberOfEpisodes
    iterated_expected_value = iterated_sum/args.numberOfEpisodes
    print("sutton_barto_expected_value = {}".format(sutton_barto_expected_value))
    print ("iterated_expected_value = {}".format(iterated_expected_value))




if __name__ == '__main__':
    main()