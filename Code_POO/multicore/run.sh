#!/bin/bash

set -o errexit
set -o xtrace

now=$(date +"%Y%m%d_%H%M%S")

plan=Head-and-Neck_15
plan_folder=~/Repo/RT25/TROTS/data/$plan

# # Adam
# folder=../../Results/Adam
# results_folder=$folder/$plan 
# result_path=$results_folder/x_$plan_$now.txt
# mkdir -p $results_folder

# #fluence_folder=~/RT25/gradient/fluences
# fluence_folder=~/Repo/RT25/TROTS/data/$plan
# fluence_prefix=x_

# make
# ./adam_mkl $plan_folder $result_path $fluence_folder $fluence_prefix


# Gardient
folder=../../Results/SGD
results_folder=$folder/$plan 
result_path=$results_folder/x_$plan_$now.txt
mkdir -p $results_folder

#fluence_folder=~/RT25/gradient/fluences
fluence_folder=~/Repo/RT25/TROTS/data/$plan
fluence_prefix=x_

make
./gradient_mkl $plan_folder $result_path $fluence_folder $fluence_prefix
