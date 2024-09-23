from models.manipulation.base_manipulation import BaseManipulation
from env.base_sapien_env import BaseEnv
from env.sapien_envs.open_cabinet import OpenCabinetEnv
from env.my_vec_env import MultiVecEnv
from utils.transform import *
from logging import Logger

import os
import time
import yaml


class OpenDrawerManipulation(BaseManipulation) :

    def __init__(self, env : MultiVecEnv, cfg : dict, logger : Logger) :

        super().__init__(env, cfg, logger)

    def plan_pathway(self, center, axis, eval=False) :

        batch_size = center.shape[0]

        x_ = np.array([[1, 0, 0]] * batch_size)
        y_ = np.array([[0, 1, 0]] * batch_size)
        z_ = np.array([[0, 0, 1]] * batch_size)

        # Computing Pre-grasp Pose
        pre_grasp_axis = axis[:, 0]
        pre_grasp_axis -= z_ * (pre_grasp_axis * z_).sum(axis=-1, keepdims=True)
        norm = np.linalg.norm(pre_grasp_axis, axis=-1, keepdims=True)
        pre_grasp_axis = np.where(norm<1e-8, y_, pre_grasp_axis / (norm+1e-8))
        pre_grasp_p = center - pre_grasp_axis * 0.2
        pre_grasp_y = -z_
        pre_grasp_z = pre_grasp_axis
        pre_grasp_x = np.cross(pre_grasp_y, pre_grasp_z)
        axis_from = np.concatenate([
            x_[:, np.newaxis, :],
            y_[:, np.newaxis, :],
            z_[:, np.newaxis, :]
        ], axis=1)
        axis_to = np.concatenate([
            pre_grasp_x[:, np.newaxis, :],
            pre_grasp_y[:, np.newaxis, :],
            pre_grasp_z[:, np.newaxis, :]
        ], axis=1)
        pre_grasp_q = batch_get_quaternion(axis_from, axis_to)
        pre_grasp_pose = np.concatenate([pre_grasp_p, pre_grasp_q], axis=-1)

        # Performing Grasp
        self.env.class_method("toggle_gripper", open=True)
        res = self.env.hand_move_to(pre_grasp_pose, time=2, wait=2, planner="path", no_collision_with_front=True)

        proceed = np.ones(batch_size, dtype=np.int32)
        grasp_p = pre_grasp_p

        if self.cfg["closed_loop"] :
            for i in range(3) :

                # Computing Grasp Pose
                grasp_p = grasp_p + pre_grasp_axis * 0.06 * proceed[..., None]
                grasp_q = pre_grasp_q
                grasp_pose = np.concatenate([grasp_p, grasp_q], axis=-1)

                res = self.env.hand_move_to(grasp_pose, time=2, wait=1, planner="ik")

                self.env.class_method("_release_target")
                error = np.linalg.norm(self.env.hand_pose()[:, :3] - grasp_p, axis=-1)
                proceed = proceed & (error < 0.01)

            grasp_p = grasp_p - pre_grasp_axis * 0.01
            grasp_q = pre_grasp_q
            grasp_pose = np.concatenate([grasp_p, grasp_q], axis=-1)
            res = self.env.hand_move_to(grasp_pose, time=2, wait=1, planner="ik")
        else :
            grasp_p = grasp_p + pre_grasp_axis * 0.18
            grasp_q = pre_grasp_q
            grasp_pose = np.concatenate([grasp_p, grasp_q], axis=-1)

            res = self.env.hand_move_to(grasp_pose, time=2, wait=1, planner="path")

            self.env.class_method("_release_target")

        self.env.class_method("toggle_gripper", open=False)

        cur_dir = - pre_grasp_axis

        dof_lists = []
        skip_list = []

        critical_dof = None
        episode_dof_list = []

        num_step_sizes = len(self.cfg["step_sizes"])
        once_SAM2 = True

        """
        # axis prediction history
        pivot_point_list = []
        axis_direction_list = []
        batch_colors_list = []
        """

        for step_idx, step_size in enumerate(self.cfg["step_sizes"]):

            print()
            print(f"[Step {step_idx}] step_size: {step_size}")

            cur_p = self.env.gripper_pose()[:, :3]
            pred_p = cur_p + cur_dir * step_size

            next_y = -z_
            next_z = -cur_dir
            next_x = np.cross(next_y, next_z)
            axis_from = np.concatenate([
                x_[:, np.newaxis, :],
                y_[:, np.newaxis, :],
                z_[:, np.newaxis, :]
            ], axis=1)
            axis_to = np.concatenate([
                next_x[:, np.newaxis, :],
                next_y[:, np.newaxis, :],
                next_z[:, np.newaxis, :]
            ], axis=1)
            pred_q = batch_get_quaternion(axis_from, axis_to)

            pred_pose = np.concatenate([pred_p, pred_q], axis=-1)

            step_dof_lists = self.env.gripper_move_to(pred_pose, time=step_size*10, wait=step_size*5,  planner="ik" if self.cfg["closed_loop"] else "path")
            for i, step_dof_list in enumerate(step_dof_lists):
                while len(dof_lists) < i + 1:
                    empty_list = []
                    dof_lists.append(empty_list)
                if len(step_dof_list) >= 3:
                    dof_lists[i] += step_dof_list[2]

            if not self.cfg["axis_prediction"]:
                # update as RGBManip

                new_p = self.env.gripper_pose()[:, :3]
                new_dir = new_p - cur_p
                new_dir[:, 2] = 0       # project to xy-plane
                new_dir = normalize(new_dir)

                delta = new_dir - cur_dir

                dot = (new_dir * cur_dir).sum(axis=-1, keepdims=True)
                dot = np.clip(dot, -1, 1)

                cur_dir = normalize(cur_dir + 2*delta*dot)
            else:
                # update guided by axis prediction

                obs = self.env.get_observation()
                episode_dof_list = obs["object_dof"][:, 0]

                if critical_dof is None:
                    critical_dof = episode_dof_list

                if (step_idx < num_step_sizes - 1) and once_SAM2:
                    if self.cfg["once_SAM2"]:
                        once_SAM2 = False
        
                    print(f"Ours begins at time_step {i}. ")
                    print(f"episode_dof_list: {episode_dof_list}")
                    
                    skip_list = self.env.get_skip_list(episode_dof_list)

                    # if (last_episode_dof_list is not None):
                    #     for i, (dof, last_dof) in enumerate(zip(episode_dof_list, last_episode_dof_list)):
                    #         if np.isclose(
                    #             dof, last_dof, 
                    #             atol = 1e-4
                    #         ):
                    #             skip_list[i] = True
                    # last_episode_dof_list = episode_dof_list
                    
                    # print(f"skip_list: {skip_list}")

                    # predict axis
                    estimated_axis_direction, estimated_pivot_point \
                        = self.env.predict_axis(
                            skip_list = skip_list
                        )

                    print(f"estimated_axis_direction: {estimated_axis_direction}")
                    print(f"estimated_pivot_point: {estimated_pivot_point}")

                    cur_p = self.env.gripper_pose()[:, : 3]
                    target_dir = normalize(estimated_axis_direction)

                    cur_dir = target_dir

                    batch_colors = [(num_step_sizes - step_idx) / num_step_sizes, 0, 0]
                    batch_colors = [batch_colors] * batch_size
                    # batch_colors_list.append(batch_colors)

                    self.env.save_axis(
                        estimated_pivot_point, 
                        estimated_axis_direction, 
                        batch_colors, 
                    )

                    """
                    pivot_point_list.append(estimated_pivot_point)
                    axis_direction_list.append(estimated_axis_direction)

                    def draw_axis(
                        estimated_pivot_point, 
                        estimated_axis_direction, 
                        batch_colors
                    ):
                        for k in np.arange(0, 2, 0.05):
                            batch_xyzs = []
                            
                            for (pivot_point, axis_direction) in zip(estimated_pivot_point, estimated_axis_direction):
                                if axis_direction[2] < 0:
                                    batch_xyzs.append(pivot_point - k * axis_direction)
                                else:
                                    batch_xyzs.append(pivot_point + k * axis_direction)

                            self.env.draw_point(
                                xyzs = batch_xyzs, 
                                colors = batch_colors
                            )

                    for estimated_pivot_point, estimated_axis_direction, batch_colors \
                        in zip(
                            pivot_point_list, 
                            axis_direction_list, 
                            batch_colors_list
                        ):
                            draw_axis(
                                estimated_pivot_point, 
                                estimated_axis_direction, 
                                batch_colors
                            )
                    """

        def save_to_yaml(
            env_idx, 
            dof_list, 
            critical_dof, 
            skipped = False
        ):
            tmp_path = os.path.join(".", "tmp")
            yamls_path = os.path.join(tmp_path, "yamls", "open_drawer")
            if not os.path.exists(yamls_path):
                os.makedirs(yamls_path)

            cur_time = time.strftime("%Y-%m-%d_%H-%M-%S")
            if skipped:
                yaml_name = f"{cur_time}_{env_idx}_skipped.yaml"
            else:
                yaml_name = f"{cur_time}_{env_idx}.yaml"
            yaml_path = os.path.join(yamls_path, yaml_name)

            res_dict = {}
            res_dict["dof_list"] = f"{dof_list}"
            res_dict["critical_dof"] = f"{critical_dof}"

            with open(yaml_path, "w") as f:
                yaml.dump(
                    res_dict, f, 
                    Dumper = yaml.SafeDumper, 
                )

        for i, dof_list in enumerate(dof_lists):
            save_to_yaml(
                env_idx = i, 
                dof_list = dof_list, 
                critical_dof = critical_dof, 
                skipped = skip_list[i] if (i < len(skip_list)) else False
            )