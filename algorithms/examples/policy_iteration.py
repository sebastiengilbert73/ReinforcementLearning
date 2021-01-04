import logging
import argparse
import ReinforcementLearning.algorithms.policy as rl_policy
import ReinforcementLearning.environments as environments
from ReinforcementLearning.environments import gridworlds
from ReinforcementLearning.environments import jacks_car_rental
from ReinforcementLearning.environments import gamblers_problem
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--randomSeed', help="The seed for the random module. Default: 0", type=int, default=0)
parser.add_argument('--environment', help="The environment to use. Default: 'gridworld1'", default='gridworld1')
parser.add_argument('--legalActionsAuthority', help="The authority that filters the legal actions. Default: 'AllActionsLegal'", default='AllActionsLegal')
parser.add_argument('--gamma', help="The discount factor. Default: 0.9", type=float, default=0.9)
#parser.add_argument('--epsilon', help="The probability of a random move, for epsilon-greedy policies. Default: 0.1", type=float, default=0.1)
parser.add_argument('--minimumChange', help="For the evaluator, the minimum value change to keep iterating. Default: 0.01", type=float, default=0.01)
parser.add_argument('--numberOfSelectionsPerState', help="For the evaluator, the number of tried selections per state. Should be 1 if the policy and the environment are deterministic. Default: 100", type=int, default=100)
parser.add_argument('--evaluatorMaximumNumberOfIterations', help="For the evaluator, the maximum number of iterations. Default: 1000", type=int, default=1000)
parser.add_argument('--initialValue', help="The initial value for all states. Default: 0", type=float, default=0)
#parser.add_argument('--numberOfTrialsPerAction', help="The number of trials per action. For deterministic environments, should be 1. Default: 1", type=int, default=1)
parser.add_argument('--iteratorMaximumNumberOfIterations', help="For the policy iterator, the maximum number of iterations. Default: 100", type=int, default=100)
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s [%(levelname)s] %(message)s')
random.seed(args.randomSeed)

def main():
    logging.info("policy_iteration.py    main()   environment={}".format(args.environment))

    # Create the environment
    # It must implement gym.Env, ActionsSet(), StatesSet(), SetState(s)
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
        raise NotImplementedError("main(): Not implemented legal actions authority '{}'".format(args.legalActionsAuthority))

    # Create the policy evaluator
    policy_evaluator = rl_policy.PolicyEvaluator(environment=environment,
                                              gamma=args.gamma,
                                              minimum_change=args.minimumChange,
                                              number_of_selections_per_state=args.numberOfSelectionsPerState,
                                              maximum_number_of_iterations=args.evaluatorMaximumNumberOfIterations,
                                              initial_value=args.initialValue
                                              )

    # Create the policy iterator
    policy_iterator = rl_policy.PolicyIterator(
         environment=environment,
         policy_evaluator=policy_evaluator,
         legal_actions_authority=legal_actions_authority,
         gamma=args.gamma,
         #epsilon=args.epsilon,
         #initial_value=args.initialValue,
         #number_of_trials_per_action=args.numberOfTrialsPerAction,
         maximum_number_of_iterations=args.iteratorMaximumNumberOfIterations,
         print_steps=True
    )

    # Iterate the policy
    policy, policy_state_to_actions_probabilities, policy_state_to_most_valuable_action = policy_iterator.IteratePolicy()

    #print("policy_state_to_actions_probabilities =\n{}".format(policy_state_to_actions_probabilities))
    print ("policy_state_to_most_valuable_action = \n{}".format(policy_state_to_most_valuable_action))

    # Write outputs to file
    WriteOutputs(policy, policy_state_to_actions_probabilities, environment)


def WriteOutputs(policy, policy_state_to_actions_probabilities, environment):
    if args.environment.lower() == 'jackscarrental':
        policy_arr = np.zeros((21, 21), dtype=float)
        for state in range(441):
            (cars_at_location1, cars_at_location2) = jacks_car_rental.JacksCarRental.NumberOfCarsAtEachLocation(state)
            action_to_probability_dict = policy_state_to_actions_probabilities[state]
            highest_probability = 0
            most_probable_action = None
            for (action, probability) in action_to_probability_dict.items():
                if probability > highest_probability:
                    highest_probability = probability
                    most_probable_action = action
            policy_arr[cars_at_location1, cars_at_location2] = most_probable_action

        output_filepath = '/tmp/JacksCarRental_policy.csv'
        with open(output_filepath, 'w') as output_file:
            for cars_at_location1 in range(21):
                for cars_at_location2 in range(21):
                    output_file.write(str(policy_arr[cars_at_location1, cars_at_location2]))
                    if cars_at_location2 != 20:
                        output_file.write(',')
                output_file.write('\n')
    elif args.environment.lower() == 'jackscarrental4.4':
        policy_arr = np.zeros((21, 21), dtype=float)
        for state in range(441):
            (cars_at_location1, cars_at_location2) = jacks_car_rental.JacksCarRental.NumberOfCarsAtEachLocation(state)
            action_to_probability_dict = policy_state_to_actions_probabilities[state]
            highest_probability = 0
            most_probable_action = None
            for (action, probability) in action_to_probability_dict.items():
                if probability > highest_probability:
                    highest_probability = probability
                    most_probable_action = action
            policy_arr[cars_at_location1, cars_at_location2] = most_probable_action

        output_filepath = '/tmp/JacksCarRental4.4_policy.csv'
        with open(output_filepath, 'w') as output_file:
            for cars_at_location1 in range(21):
                for cars_at_location2 in range(21):
                    output_file.write(str(policy_arr[cars_at_location1, cars_at_location2]))
                    if cars_at_location2 != 20:
                        output_file.write(',')
                output_file.write('\n')
    elif args.environment.lower() == 'gamblersproblem':
        # Evaluate precisely
        policy_evaluator = rl_policy.PolicyEvaluator(environment=environment,
                                                     gamma=args.gamma,
                                                     minimum_change=0.001,
                                                     number_of_selections_per_state=1,
                                                     maximum_number_of_iterations=100,
                                                     initial_value=0
                                                     )
        (state_to_value_dict, change, completed_iterations) = policy_evaluator.Evaluate(policy)

        output_filepath = '/tmp/GamblersProblem_policy.csv'
        with open(output_filepath, 'w') as output_file:
            for origin_state in range(1, 100):
                stake = policy.Select(origin_state)
                value = state_to_value_dict[origin_state]
                output_file.write("{},{},{}\n".format(origin_state, stake, value))

if __name__ == '__main__':
    main()