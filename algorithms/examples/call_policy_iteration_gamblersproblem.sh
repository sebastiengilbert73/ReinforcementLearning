#!/bin/bash
python policy_iteration.py \
	--randomSeed=0 \
	--environment=GamblersProblem \
	--legalActionsAuthority=GamblersPossibleStakes \
	--gamma=0.9 \
	--epsilon=0.1 \
	--minimumChange=0.05 \
	--numberOfSelectionsPerState=1 \
	--evaluatorMaximumNumberOfIterations=50 \
	--initialValue=1.0 \
	--numberOfTrialsPerAction=1 \
	--iteratorMaximumNumberOfIterations=100 \