import os
import shutil

import yaml

import numpy as np


def load_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

def save_yaml(
    yaml_object, 
    save_path, 
    yaml_name
):
    yaml_path = os.path.join(save_path, yaml_name)

    with open(yaml_path, "w") as f:
        yaml.dump(
            yaml_object, 
            f, 
            Dumper = yaml.SafeDumper
        )

def get_yaml_item(
    yaml_object, 
    key
):
    return yaml_object[key]

def write_yaml_object(
    yaml_object, 
    key, val
):
    yaml_object[key] = val





