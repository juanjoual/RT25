#!/bin/bash

set -o errexit
set -o xtrace

now=$(date +"%Y%m%d_%H%M%S")

plan=Head-and-Neck_01
plan_folder=~/Repo/RT25/TROTS/data/$plan

# Adam
results_folder=results_adam
result_path=$results_folder/x_$plan_$now.txt
mkdir -p $results_folder

#fluence_folder=~/RT25/gradient/fluences
fluence_folder=~/Repo/RT25/TROTS/data/$plan
fluence_prefix=x_

make
./adam_mkl $plan_folder $result_path $fluence_folder $fluence_prefix


# # Gardient
# results_folder=results_gradient
# result_path=$results_folder/x_$plan_$now.txt
# mkdir -p $results_folder

# #fluence_folder=~/RT25/gradient/fluences
# fluence_folder=~/Repo/RT25/TROTS/data/$plan
# fluence_prefix=x_

# make
# ./gradient_mkl $plan_folder $result_path $fluence_folder $fluence_prefix
