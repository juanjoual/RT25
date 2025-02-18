#!/bin/bash

set -o errexit
set -o xtrace

now=$(date +"%Y%m%d_%H%M%S")

plan=3
plan=$1
plan_folder=~/Repo/RT25/NIO/$plan

# Adam
folder=../../Results/Adam
results_folder=$folder/$plan 
result_path=$results_folder/x_$plan_$now.txt
mkdir -p $results_folder

fluence_folder=~/RT25/NIO/$plan
fluence_prefix=x_

make
./adam_mkl $plan_folder $result_path $fluence_folder $fluence_prefix


# # Gardient
# folder=../../Results/SGD
# results_folder=$folder/$plan 
# result_path=$results_folder/x_$plan_$now.txt
# mkdir -p $results_folder

# fluence_folder=~/RT25/NIO/$plan
# fluence_prefix=x_

# make
# ./gradient_mkl $plan_folder $result_path $fluence_folder $fluence_prefix
