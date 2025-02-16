#!/bin/bash

set -o errexit
set -o xtrace

now=$(date +"%Y%m%d_%H%M%S")

plan=Head-and-Neck_02
plan_folder=~/RT25/TROTS/data/$plan

results_folder=results
result_path=$results_folder/x_$plan_$now.txt
mkdir -p $results_folder

#fluence_folder=~/RT25/gradient/fluences
fluence_folder=~/RT25/TROTS/data/$plan
fluence_prefix=x_

make
./gradient_mkl $plan_folder $result_path $fluence_folder $fluence_prefix
