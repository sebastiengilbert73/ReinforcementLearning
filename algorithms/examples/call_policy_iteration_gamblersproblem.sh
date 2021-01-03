#!/bin/bash
python policy_iteration.py \
	--randomSeed=0 \
	--environment=GamblersProblem \
	--legalActionsAuthority=GamblersPossibleStakes \
	--gamma=1.0 \
	--minimumChange=0.001 \
	--numberOfSelectionsPerState=1 \
	--evaluatorMaximumNumberOfIterations=100 \
	--initialValue=0 \
	--iteratorMaximumNumberOfIterations=1000 \