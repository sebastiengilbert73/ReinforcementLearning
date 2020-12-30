#!/bin/bash
python policy_iteration.py \
	--randomSeed=0 \
	--environment=JacksCarRental \
	--legalActionsAuthority=JacksPossibleMoves \
	--gamma=0.9 \
	--minimumChange=10 \
	--numberOfSelectionsPerState=1 \
	--evaluatorMaximumNumberOfIterations=100 \
	--initialValue=0 \
	--numberOfTrialsPerAction=10 \
	--iteratorMaximumNumberOfIterations=100 \