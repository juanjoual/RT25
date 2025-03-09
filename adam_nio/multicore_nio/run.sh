#!/bin/bash

set -o errexit
set -o xtrace

now=$(date +"%Y%m%d_%H%M%S")

plan=5
# plan=$1
plan_folder=~/Repo/RT25/NIO/$plan

results_folder=results
result_path=$results_folder/x_$plan_$now.txt
mkdir -p $results_folder

#fluence_folder=~/RT25/gradient/fluences
fluence_folder=~/Repo/RT25/NIO/$plan
fluence_prefix=x_

make
./gradient_mkl $plan_folder $result_path $fluence_folder $fluence_prefix
