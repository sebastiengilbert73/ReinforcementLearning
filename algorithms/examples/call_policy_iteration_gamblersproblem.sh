#!/bin/bash
python policy_iteration.py \
	--randomSeed=0 \
	--environment=GamblersProblem \
	--legalActionsAuthority=GamblersPossibleStakes \
	--gamma=0.9 \
	--minimumChange=0.01 \
	--numberOfSelectionsPerState=1 \
	--evaluatorMaximumNumberOfIterations=50 \
	--initialValue=1.0 \
	--iteratorMaximumNumberOfIterations=100 \