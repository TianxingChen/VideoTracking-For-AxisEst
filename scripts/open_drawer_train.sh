#!/bin/bash

gpu_id=${1}
export CUDA_VISIBLE_DEVICES=${gpu_id}

OPEN_DRAWER_TRAIN="python train.py dataset=drawer_train task=open_drawer pose_estimator=adapose_drawer manipulation=open_drawer controller=heuristic_pose train=test task.num_envs=1"

num_round=${2:-1}

for i in $(seq 1 $num_round); do
    echo "Running $i-th OPEN_DRAWER_TRAIN ..."

    $OPEN_DRAWER_TRAIN
done
