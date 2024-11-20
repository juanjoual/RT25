#!/bin/bash

set -o errexit
set -o xtrace

now=$(date +"%Y%m%d_%H%M%S")

plan=5
plan_folder=~/RT25/plans/$plan

results_folder=results
result_path=$results_folder/x_$plan_$now.txt
mkdir -p $results_folder

fluence_folder=~/RT25/gradient/fluences
fluence_prefix=x_5_start_0001

n_plans=1

make
./gradient_mkl $plan $plan_folder $result_path $fluence_folder $fluence_prefix $n_plans
