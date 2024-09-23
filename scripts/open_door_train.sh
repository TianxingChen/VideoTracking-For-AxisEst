#!/bin/bash

gpu_id=${1}
export CUDA_VISIBLE_DEVICES=${gpu_id}

OPEN_CABINET_TRAIN="python train.py dataset=cabinet_train task=open_cabinet pose_estimator=adapose_cabinet manipulation=open_cabinet controller=heuristic_pose train=test task.num_envs=1"

num_round=${2:-1}

for i in $(seq 1 $num_round); do
    echo "Running $i-th OPEN_CABINET_TRAIN ..."

    $OPEN_CABINET_TRAIN
done
