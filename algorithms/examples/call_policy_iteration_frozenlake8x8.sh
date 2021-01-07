#!/bin/bash
python policy_iteration.py \
	--randomSeed=0 \
	--environment=FrozenLake8x8 \
	--legalActionsAuthority=AllActionsLegal \
	--gamma=0.9 \
	--minimumChange=0.01 \
	--evaluatorMaximumNumberOfIterations=10 \
	--initialValue=0 \
	--iteratorMaximumNumberOfIterations=100 \