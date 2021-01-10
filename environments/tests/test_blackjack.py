import logging
import argparse
import ReinforcementLearning.algorithms.policy as rl_policy
import ReinforcementLearning.algorithms.monte_carlo as monte_carlo
import ReinforcementLearning.environments as environments
from ReinforcementLearning.environments import blackjack
import random

parser = argparse.ArgumentParser()
parser.add_argument('--randomSeed', help="The seed for the random module. Default: 0", type=int, default=0)
parser.add_argument('--policy', help="The policy. Default: 'random'", default='random')
parser.add_argument('--gamma', help="The discount factor. Default: 1.0", type=float, default=1.0)
parser.add_argument('--numberOfEpisodes', help="The number of episodes. Default: 20", type=int, default=20)
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s [%(levelname)s] %(message)s')
random.seed(args.randomSeed)

def main():
    logging.info("test_blackjack.py main()")
    environment = blackjack.Blackjack()

    policy_evaluator = monte_carlo.FirstVisitPolicyEvaluator(
        environment=environment,
        gamma=args.gamma,
        number_of_iterations=1000,
        initial_value=0,
        episode_maximum_length=1000
    )

    # Create the authority to filter the legal actions
    actions_set = environment.ActionsSet()
    legal_actions_authority =  rl_policy.AllActionsLegalAuthority(actions_set)

    policy = None
    if args.policy.lower() == 'random':
        policy = rl_policy.Random(legal_actions_authority)
    elif args.policy.lower() == 'hitupto19':
        policy = blackjack.HitUpTo(19)
    else:
        raise NotImplementedError("main(): Not implemented policy {}".format(args.policy))

    for episodeNdx in range(args.numberOfEpisodes):
        episode = policy_evaluator.Episode(policy)
        for (observation, reward) in episode:
            obsTuple = environment.TupleFromState(observation)
            print ("{}, reward = {}".format(obsTuple, reward), end=' ')
        print ()
        discounted_return = policy_evaluator.Return(episode)
        print ("discounted_return = {}".format(discounted_return))

if __name__ == '__main__':
    main()