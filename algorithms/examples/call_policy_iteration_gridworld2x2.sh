#!/bin/bash
python policy_iteration.py \
	--randomSeed=0 \
	--environment=gridworld2x2 \
	--legalActionsAuthority=AllActionsLegal \
	--gamma=0.9 \
	--minimumChange=0.1 \
	--numberOfSelectionsPerState=1 \
	--evaluatorMaximumNumberOfIterations=3000 \
	--initialValue=0 \
	--iteratorMaximumNumberOfIterations=100 \