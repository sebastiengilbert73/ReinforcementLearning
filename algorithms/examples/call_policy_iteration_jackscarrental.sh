#!/bin/bash
python policy_iteration.py \
	--randomSeed=0 \
	--environment=JacksCarRental \
	--legalActionsAuthority=JacksPossibleMoves \
	--gamma=0.9 \
	--epsilon=0.1 \
	--minimumChange=50 \
	--numberOfSelectionsPerState=1 \
	--evaluatorMaximumNumberOfIterations=1000 \
	--initialValue=0 \
	--numberOfTrialsPerAction=1 \
	--iteratorMaximumNumberOfIterations=100 \