#!/bin/bash
python policy_iteration.py \
	--randomSeed=0 \
	--environment=JacksCarRental4.4 \
	--legalActionsAuthority=JacksPossibleMoves \
	--gamma=0.9 \
	--minimumChange=50 \
	--evaluatorMaximumNumberOfIterations=10 \
	--initialValue=0 \
	--iteratorMaximumNumberOfIterations=100 \