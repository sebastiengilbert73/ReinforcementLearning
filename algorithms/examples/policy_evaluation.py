import logging
import argparse
import ReinforcementLearning.algorithms.policy as policy
import ReinforcementLearning.environments as environments
from ReinforcementLearning.environments import gridworld1
import random

parser = argparse.ArgumentParser()
parser.add_argument('--randomSeed', help="The seed for the random module. Default: 0", type=int, default=0)
parser.add_argument('--environment', help="The environment to use. Default: 'gridworld1'", default='gridworld1')
parser.add_argument('--legalActionsAuthority', help="The authority that filters the legal actions. Default: 'AllActionsLegal'", default='AllActionsLegal')
parser.add_argument('--policy', help="The policy. Default: 'random'", default='random')
parser.add_argument('--gamma', help="The discount factor. Default: 0.9", type=float, default=0.9)
parser.add_argument('--minimumChange', help="The minimum value change to keep iterating. Default: 0.01", type=float, default=0.01)
parser.add_argument('--numberOfSelectionsPerState', help="The number of tried selections per state. Should be 1 for deterministic policies. Default: 100", type=int, default=100)
parser.add_argument('--maximumNumberOfIterations', help="The maximum number of iterations. Default: 1000", type=int, default=1000)
parser.add_argument('--initialValue', help="The initial value for all states. Default: 0", type=float, default=0)
parser.add_argument('--epsilon', help="For epsilon-greedy policies, the probability of choosing a random action. Default: 0.1", type=float, default=0.1)
parser.add_argument('--numberOfTrialsPerAction', help="The number of trials per action. For deterministic environments, should be 1. Default: 1", type=int, default=1)
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s [%(levelname)s] %(message)s')
random.seed(args.randomSeed)

def main():
    logging.info("policy_evaluation.py    main()   environment={}".format(args.environment))

    # Create the environment
    # It must implement gym.Env, ActionsSet(), StatesSet(), SetState(s)
    environment = None
    if args.environment.lower() == 'gridworld1':
        environment = gridworld1.GridWorld1()
    elif args.environment.lower() == 'gridworld2x2':
        environment = gridworld1.GridWorld2x2()
    else:
        raise NotImplementedError("main(): Not implemented environment '{}'".format(args.environment))

    # Create the authority to filter the legal actions
    actions_set = environment.ActionsSet()
    legal_actions_authority = None
    if args.legalActionsAuthority.lower() == 'allactionslegal':
        legal_actions_authority = policy.AllActionsLegalAuthority(actions_set)
    else:
        raise NotImplementedError("main(): Not implemented legal actions authority '{}'".format(args.legalActionsAuthority))

    # Create the evaluated policy
    evaluated_policy = None
    if args.policy.lower() == 'random':
        evaluated_policy = policy.Random(legal_actions_authority)
    elif args.policy.lower() == 'epsilongreedy':
        states_set = environment.StatesSet()
        state_to_value_dict = {s: args.initialValue for s in states_set}
        evaluated_policy = policy.EpsilonGreedy(
            state_to_value_dict=state_to_value_dict,
            legal_actions_authority=legal_actions_authority,
            environment=environment,
            epsilon=args.epsilon,
            gamma=args.gamma,
            number_of_trials_per_action=args.numberOfTrialsPerAction
        )
    else:
        raise NotImplementedError("main(): Not implemented policy '{}'".format(args.policy))

    # Create the policy evaluator
    policy_evaluator = policy.PolicyEvaluator(environment=environment,
                                              gamma=args.gamma,
                                              minimum_change=args.minimumChange,
                                              number_of_selections_per_state=args.numberOfSelectionsPerState,
                                              maximum_number_of_iterations=args.maximumNumberOfIterations,
                                              initial_value=args.initialValue
                                              )

    # Evaluation
    logging.info("Starting evaluation...")
    state_to_value_dict = policy_evaluator.Evaluate(evaluated_policy)
    logging.info("Done!")
    print("state_to_value_dict = \n{}".format(state_to_value_dict))

if __name__ == '__main__':
    main()