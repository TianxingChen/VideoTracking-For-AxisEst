import sys
import getopt

import os
import shutil

import yaml

import numpy as np

from yaml_utils import *


result_yamls_path = os.path.join(".", "result_yamls")
task_path = "/home/vcc/Desktop/ManiArticulation/tmp/yamls/open_cabinet"


def extract_dof_list(yaml_object):
    return eval(
        get_yaml_item(yaml_object, "dof_list")
    )

def get_dof_list_list(
    task_path, 
    use_skipped = False
):
    dof_list_list = []

    for yaml_name in os.listdir(task_path):
        if use_skipped and (yaml_name[-13 : -5] == "_skipped"):
            continue

        yaml_path = os.path.join(task_path, yaml_name)
        yaml_object = load_yaml(yaml_path)

        dof_list = extract_dof_list(yaml_object)
        dof_list_list.append(dof_list)
    
    return dof_list_list

def get_max_dof_list(dof_list_list):
    max_dof_list = [
        max(dof_list) for dof_list in dof_list_list
    ]

    return max_dof_list

use_skipped = False
train = False

dof_list_list = []
max_dof_list = []

def save_as_yaml(
    yaml_object, 
    success_rate_list, 
    yaml_name
):
    if train or ("train" in task_path):
        write_yaml_object(
            yaml_object, 
            key = "Ours_train_success_rate_list", 
            val = f"{success_rate_list}"
        )
    else:
        write_yaml_object(
            yaml_object, 
            key = "Ours_test_success_rate_list", 
            val = f"{success_rate_list}"
        )

    save_yaml(
        yaml_object, 

        save_path = result_yamls_path, 
        yaml_name = yaml_name
    )

def cal_success_rate(
    dof_threshold, 
    use_max = False, 

    max_num_exps = None
):
    num_exps = len(dof_list_list)
    if max_num_exps is not None:
        num_exps = min(num_exps, max_num_exps)

    true_num_exps = 0
    num_success = 0

    for i in range(1, num_exps, 1):
        true_num_exps += 1

        if use_max:
            num_success += (max_dof_list[i] >= dof_threshold)
        else:
            num_success += (dof_list_list[i][-1] >= dof_threshold)

    assert (true_num_exps != 0)
    return true_num_exps, num_success, num_success / true_num_exps

def cal_open_drawer():
    yaml_path = os.path.join(result_yamls_path, "open_drawer_success_rate.yaml")
    yaml_object = load_yaml(yaml_path)

    success_rate_list = []

    dof_threshold_list = eval(
        get_yaml_item(
            yaml_object, 
            key = "open_drawer_dof_list"
        )
    )
    dof_threshold_list = [
        dof_threshold / 100 \
            for dof_threshold in dof_threshold_list
    ]  # cm -> m
    
    for dof_threshold in dof_threshold_list:
        num_exps, num_success, success_rate = cal_success_rate(
            dof_threshold = dof_threshold, 
            use_max = True, 
            # use_max = False, 
        )

        success_rate *= 100

        success_rate_list.append(success_rate)

        print(f"[dof_threshold] {dof_threshold}")
        print(f"success_rate: {success_rate:.1f}% ({num_success} / {num_exps})")
        print()
    
    if not use_skipped:
        save_as_yaml(
            yaml_object = yaml_object, 
            success_rate_list = success_rate_list, 
            yaml_name = "open_drawer_success_rate.yaml"
        )

def cal_open_door():
    yaml_path = os.path.join(result_yamls_path, "open_door_success_rate.yaml")
    yaml_object = load_yaml(yaml_path)

    use_degree = True

    success_rate_list = []

    def threshold_in_degree():
        dof_threshold_list = eval(
            get_yaml_item(
                yaml_object, 
                key = "open_door_dof_list"
            )
        )

        for i, dof_threshold in enumerate(dof_threshold_list):
            dof_threshold = np.radians(dof_threshold)

            num_exps, num_success, success_rate = cal_success_rate(
                dof_threshold = dof_threshold, 
                use_max = True, 
                # use_max = False,
            )

            success_rate *= 100

            success_rate_list.append(success_rate)

            print(f"[dof_threshold] {dof_threshold} ({dof_threshold_list[i]} deg)")
            print(f"success_rate: {success_rate:.1f}% ({num_success} / {num_exps})")
            print()

    if use_degree:
        threshold_in_degree()
    else:
        pass

    if not use_skipped:
        save_as_yaml(
            yaml_object = yaml_object, 
            success_rate_list = success_rate_list, 
            yaml_name = "open_door_success_rate.yaml"
        )

if __name__ == "__main__":
    opts, args = getopt.getopt(
        sys.argv[1 :], 
        "p:st", 
        ["path="]
    )

    for opt, arg in opts:
        if opt in ("-p", "--path"):
            task_path = arg
        elif opt in ("-s", "--skipped"):
            use_skipped = True
        elif opt in ("-t", "--train"):
            train = True

    print(f"task_path: {task_path}")
    dof_list_list = get_dof_list_list(
        task_path, 
        use_skipped = use_skipped, 
    )
    max_dof_list = get_max_dof_list(dof_list_list)

    if "open_drawer" in task_path:
        cal_open_drawer()
    else:
        cal_open_door()
