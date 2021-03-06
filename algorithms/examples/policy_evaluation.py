import logging
import argparse
import ReinforcementLearning.algorithms.policy as policy
import ReinforcementLearning.algorithms.dp_iteration as dp_iteration
import ReinforcementLearning.environments as environments
from ReinforcementLearning.environments import gridworlds
from ReinforcementLearning.environments import jacks_car_rental
from ReinforcementLearning.environments import gamblers_problem
import random

parser = argparse.ArgumentParser()
parser.add_argument('--randomSeed', help="The seed for the random module. Default: 0", type=int, default=0)
parser.add_argument('--environment', help="The environment to use. Default: 'gridworld1'", default='gridworld1')
parser.add_argument('--legalActionsAuthority', help="The authority that filters the legal actions. Default: 'AllActionsLegal'", default='AllActionsLegal')
parser.add_argument('--policy', help="The policy. Default: 'random'", default='random')
parser.add_argument('--gamma', help="The discount factor. Default: 0.9", type=float, default=0.9)
parser.add_argument('--minimumChange', help="The minimum value change to keep iterating. Default: 0.01", type=float, default=0.01)
parser.add_argument('--numberOfSelectionsPerState', help="The number of tried selections per state. Should be 1 if the policy and the environment are deterministic. Default: 100", type=int, default=100)
parser.add_argument('--maximumNumberOfIterations', help="The maximum number of iterations. Default: 1000", type=int, default=1000)
parser.add_argument('--initialValue', help="The initial value for all states. Default: 0", type=float, default=0)
parser.add_argument('--epsilon', help="For epsilon-greedy policies, the probability of choosing a random action. Default: 0.1", type=float, default=0.1)
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s [%(levelname)s] %(message)s')
random.seed(args.randomSeed)

def main():
    logging.info("policy_evaluation.py    main()   environment={}".format(args.environment))

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
        raise NotImplementedError("main(): Not implemented legal actions authority '{}'".format(args.legalActionsAuthority))

    # Create the evaluated policy
    states_set = environment.StatesSet()
    evaluated_policy = None
    if args.policy.lower() == 'random':
        evaluated_policy = policy.Random(legal_actions_authority)
    elif args.policy.lower() == 'epsilongreedy':
        state_to_value_dict = {s: args.initialValue for s in states_set}
        evaluated_policy = policy.EpsilonGreedy(
            state_to_value_dict=state_to_value_dict,
            legal_actions_authority=legal_actions_authority,
            environment=environment,
            epsilon=args.epsilon,
            gamma=args.gamma
        )
    elif args.policy.lower() == 'gridworld1optimalpolicy':
        evaluated_policy = gridworlds.GridWorld1OptimalPolicy()
    elif args.policy.lower() == 'gridworld2x2optimalpolicy':
        evaluated_policy = gridworlds.GridWorld2x2OptimalPolicy()
    elif args.policy.lower() == 'greedy':
        evaluated_policy = policy.Greedy({s: 0 for s in states_set}, legal_actions_authority)
    else:
        raise NotImplementedError("main(): Not implemented policy '{}'".format(args.policy))

    # Create the policy evaluator
    policy_evaluator = dp_iteration.PolicyEvaluator(environment=environment,
                                              gamma=args.gamma,
                                              minimum_change=args.minimumChange,
                                              number_of_selections_per_state=args.numberOfSelectionsPerState,
                                              maximum_number_of_iterations=args.maximumNumberOfIterations,
                                              initial_value=args.initialValue
                                              )

    # Evaluation
    logging.info("Starting evaluation...")
    (state_to_value_dict, last_change_magnitude, completed_iterations) = policy_evaluator.Evaluate(evaluated_policy)
    logging.info("Done!")
    print ("last_change_magnitude = {}; completed_iterations = {}".format(last_change_magnitude, completed_iterations))
    print("state_to_value_dict = \n{}".format(state_to_value_dict))

if __name__ == '__main__':
    main()