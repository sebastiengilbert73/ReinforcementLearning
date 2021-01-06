#!/bin/bash
python policy_iteration.py \
	--randomSeed=0 \
	--environment=JacksCarRental \
	--legalActionsAuthority=JacksPossibleMoves \
	--gamma=0.9 \
	--minimumChange=50 \
	--numberOfSelectionsPerState=1 \
	--evaluatorMaximumNumberOfIterations=10 \
	--initialValue=0 \
	--iteratorMaximumNumberOfIterations=100 \