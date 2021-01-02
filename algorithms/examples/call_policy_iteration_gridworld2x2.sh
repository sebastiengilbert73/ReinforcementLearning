#!/bin/bash
python policy_iteration.py \
	--randomSeed=0 \
	--environment=gridworld2x2 \
	--legalActionsAuthority=AllActionsLegal \
	--gamma=0.9 \
	--epsilon=0.1 \
	--minimumChange=0.1 \
	--numberOfSelectionsPerState=1 \
	--evaluatorMaximumNumberOfIterations=3000 \
	--initialValue=0 \
	--numberOfTrialsPerAction=1 \
	--iteratorMaximumNumberOfIterations=100 \