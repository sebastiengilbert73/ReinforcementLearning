#!/bin/bash
python policy_iteration.py \
	--randomSeed=0 \
	--environment=gridworld1 \
	--legalActionsAuthority=AllActionsLegal \
	--gamma=0.9 \
	--minimumChange=0.01 \
	--numberOfSelectionsPerState=1 \
	--evaluatorMaximumNumberOfIterations=1000 \
	--initialValue=0 \
	--numberOfTrialsPerAction=1 \
	--iteratorMaximumNumberOfIterations=100 \