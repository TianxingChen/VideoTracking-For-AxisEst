#!/bin/bash

gpu_id=${1}
export CUDA_VISIBLE_DEVICES=${gpu_id}

OPEN_CABINET_TRAIN="python train.py dataset=cabinet_train task=open_cabinet pose_estimator=adapose_cabinet manipulation=open_cabinet controller=heuristic_pose train=test task.num_envs=1"
OPEN_CABINET_TEST="python train.py dataset=cabinet_test task=open_cabinet pose_estimator=adapose_cabinet manipulation=open_cabinet controller=heuristic_pose train=test task.num_envs=1"
OPEN_DRAWER_TRAIN="python train.py dataset=drawer_train task=open_drawer pose_estimator=adapose_drawer manipulation=open_drawer controller=heuristic_pose train=test task.num_envs=1"
OPEN_DRAWER_TEST="python train.py dataset=drawer_test task=open_drawer pose_estimator=adapose_drawer manipulation=open_drawer controller=heuristic_pose train=test task.num_envs=1"

num_video=${2:-1}

for i in $(seq 1 $num_video); do
    echo "Running $i-th OPEN_CABINET_TRAIN ..."

    $OPEN_CABINET_TRAIN
done

for i in $(seq 1 $num_video); do
    echo "Running $i-th OPEN_CABINET_TEST ..."

    $OPEN_CABINET_TEST
done

for i in $(seq 1 $num_video); do
    echo "Running $i-th OPEN_DRAWER_TRAIN ..."

    $OPEN_DRAWER_TRAIN
done

for i in $(seq 1 $num_video); do
    echo "Running $i-th OPEN_DRAWER_TEST ..."

    $OPEN_DRAWER_TEST
done




