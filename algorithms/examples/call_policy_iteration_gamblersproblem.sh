#!/bin/bash
python policy_iteration.py \
	--randomSeed=0 \
	--environment=GamblersProblem \
	--legalActionsAuthority=GamblersPossibleStakes \
	--gamma=1.0 \
	--minimumChange=0.000001 \
	--numberOfSelectionsPerState=1 \
	--evaluatorMaximumNumberOfIterations=1 \
	--initialValue=0.0 \
	--iteratorMaximumNumberOfIterations=150 \