#!/bin/bash
python policy_iteration.py \
	--randomSeed=0 \
	--environment=JacksCarRental \
	--legalActionsAuthority=JacksPossibleMoves \
	--gamma=0.9 \
	--epsilon=0.1 \
	--minimumChange=5 \
	--numberOfSelectionsPerState=300 \
	--evaluatorMaximumNumberOfIterations=1000 \
	--initialValue=0 \
	--numberOfTrialsPerAction=100 \
	--iteratorMaximumNumberOfIterations=100 \