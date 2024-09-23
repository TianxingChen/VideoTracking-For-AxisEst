import multiprocessing as mp
import os
from collections import defaultdict
from copy import deepcopy
from functools import partial
from multiprocessing.connection import Connection
from typing import Callable, Dict, List, Optional, Sequence, Type, Union
from utils.tools import *

import gym
import numpy as np
import sapien.core as sapien
from gym import spaces
from gym.vector.utils.shared_memory import *
from functools import partial

try:
    import torch
except ImportError:
    raise ImportError("To use ManiSkill2 VecEnv, please install PyTorch first.")

import utils.logger as logger

def _worker(
    rank: int,
    remote: Connection,
    parent_remote: Connection,
    env_class: object,
):
    # NOTE(jigu): Set environment variables for ManiSkill2
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    parent_remote.close()

    try:
        env = env_class()
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, done, info = env.step(action=data[0], drive_mode=data[1], quite=data[2])
                remote.send((obs, reward, done, info))
            elif cmd == "get_state":
                state = env.get_state()
                remote.send(state)
            elif cmd == "reset":
                # obs = env.reset()
                obs = env.reset(save_video = data)
                remote.send(obs)
            elif cmd == "load":
                env.load(data)
                remote.send(None)
            elif cmd == "close":
                remote.close()
                break
            elif cmd == "get_image" :
                obs = env.get_image(data)
                remote.send(obs)
            elif cmd == "get_observation" :
                obs = env.get_observation(data)
                remote.send(obs)
            elif cmd == "hand_move_to" :
                res = env.hand_move_to(data[0], data[1], data[2], data[3], data[4], data[5], data[6])
                remote.send(res)
            elif cmd == "cam_move_to" :
                res = env.cam_move_to(data[0], data[1], data[2], data[3], data[4], data[5], data[6])
                remote.send(res)
            # elif cmd == "gripper_move_to" :
            #     res = env.gripper_move_to(data[0], data[1], data[2], data[3], data[4], data[5], data[6])
            #     remote.send(res)
            elif cmd == "gripper_move_to_1" :
                res = env.gripper_move_to_1(data[0], data[1], data[2], data[3], data[4], data[5], data[6])
                remote.send(res)
            elif cmd == "rbot_qpos" :
                res = env.robot_qpos()
                remote.send(res)
            elif cmd == "gripper_pose" :
                obs = env.gripper_pose(data)
                remote.send(obs)
            elif cmd == "camera_pose" :
                obs = env.camera_pose(data)
                remote.send(obs)
            elif cmd == "hand_pose" :
                obs = env.hand_pose(data)
                remote.send(obs)
            elif cmd == "robot_pose" :
                pose = getattr(env, "robot_root_pose")
                remote.send(np.concatenate([pose.p, pose.q]))
            elif cmd == "class_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == "handshake":
                remote.send(None)

            elif cmd == "save_obs_list":
                env.save_obs_list()
                remote.send(None)
            elif cmd == "draw_point" :
                obs = env._draw_point(xyz = data[0], color = data[1], ret = False)
                remote.send(obs)
            elif cmd == "clear_point" :
                obs = env._clear_point()
                remote.send(obs)
            elif cmd == "determine_skip":
                obs = env.determine_skip(data)
                remote.send(obs)
            elif cmd == "predict_axis" :
                obs = env.predict_axis(data)
                remote.send(obs)
            elif cmd == "save_axis" :
                obs = env.save_axis(data[0], data[1], data[2])
                remote.send(obs)
            elif cmd == "draw_line" :
                obs = env._draw_line(
                    st_point = data[0], ed_point = data[1], 
                    color = data[2], 
                    ret = False
                )
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
    except KeyboardInterrupt:
        logger.log.info("Worker KeyboardInterrupt")
    except EOFError:
        logger.log.info("Worker EOF")
    except Exception as err:
        logger.log.error(err, exc_info=1)
    finally:
        env.close()

# Define a simple interface for RL
class MultiVecEnv() :

    def __init__(self, env_class_list, start_method: Optional[str] = None) :

        super().__init__()

        self.max_episode_length = 256
        self.exp_name = "RL test"
        self.num_envs = 1

        self.waiting = False
        self.closed = False

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        # ---------------------------------------------------------------------------- #
        # Acquire observation space to construct buffer
        # NOTE(jigu): Use a separate process to avoid creating sapien resources in the main process
        remote, work_remote = ctx.Pipe()
        args = (0, work_remote, remote, env_class_list[0])
        process = ctx.Process(target=_worker, args=args, daemon=True)
        process.start()
        work_remote.close()
        remote.send(("get_attr", "observation_space"))
        self.observation_space: spaces.Dict = remote.recv()
        remote.send(("get_attr", "state_space"))
        self.state_space: spaces.Dict = remote.recv()
        remote.send(("get_attr", "action_space"))
        self.action_space: spaces.Space = remote.recv()
        remote.send(("close", None))
        remote.close()
        process.join()

        n_envs = len(env_class_list)
        self.num_envs = n_envs

        # Initialize workers
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for rank in range(n_envs):
            work_remote = self.work_remotes[rank]
            remote = self.remotes[rank]
            args = (rank, work_remote, remote, env_class_list[rank])
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(
                target=_worker, args=args, daemon=True
            )  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

        # To make sure environments are initialized in all workers
        for remote in self.remotes:
            remote.send(("handshake", None))
        for remote in self.remotes:
            remote.recv()
    
    def _get_indices(self, indices) -> List[int]:
        """
        Convert a flexibly-typed reference to environment indices to an implied list of indices.

        :param indices: refers to indices of envs.
        :return: the implied list of indices.
        """
        if indices is None:
            indices = list(range(self.num_envs))
        elif isinstance(indices, int):
            indices = [indices]
        elif isinstance(indices, torch.Tensor) :
            indices = indices.nonzero(as_tuple=True)[0].tolist()
        elif isinstance(indices, np.ndarray) :
            indices = indices.nonzero()[0].tolist()
        else :
            raise ValueError(f"Indices must be integers, array, tensor or None, but got {indices}")
        return indices
    
    def _get_target_remotes(self, indices) -> List[Connection]:
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: refers to indices of envs.
        :return: Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]
    
    def reset_async(self, indices = None, save_video = False):
        remotes = self._get_target_remotes(indices)
        for remote in remotes:
            remote.send(("reset", save_video))
        self.waiting = True

    def reset_wait(self, indices=None):
        remotes = self._get_target_remotes(indices)
        results = [remote.recv() for remote in remotes]
        self.waiting = False
        vec_obs = merge_obs(results)
        return vec_obs

    def reset(self, indices = None, save_video = False):
        self.reset_async(indices = indices, save_video = save_video)
        return self.reset_wait(indices = indices)
    
    def step_async(self, actions: np.ndarray, drive_modes, quites) -> None:
        for remote, action, mode, quite in zip(self.remotes, actions, drive_modes, quites):
            remote.send(("step", (action, mode, quite)))
        self.waiting = True

    def step_wait(self, quite):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        if quite :
            return None
        obs_list, rews, dones, infos = zip(*results)
        vec_obs = merge_obs(obs_list)
        return vec_obs, np.asarray(rews), np.asarray(dones), infos

    def step(self, actions, drive_mode="delta", quite=False):
        if not isinstance(drive_mode, list) :
            drive_mode = [drive_mode] * self.num_envs
        if not isinstance(quite, list) :
            quite = [quite] * self.num_envs
        self.step_async(actions, drive_mode, quite)
        return self.step_wait(quite)
    
    def state_async(self) :
        for remote in self.remotes:
            remote.send(("get_state", None))
        self.waiting = True
    
    def state_wait(self) :
        state_list = [remote.recv() for remote in self.remotes]
        self.waiting = False
        vec_obs = merge_obs(state_list)
        return vec_obs
    
    def get_state(self) :
        self.state_async()
        return self.state_wait()
    
    def get_image_async(self, mask) :
        for remote in self.remotes:
            remote.send(("get_image", mask))
        self.waiting = True

    def get_image_wait(self) :
        image_list = [remote.recv() for remote in self.remotes]
        self.waiting = False
        vec_obs = merge_obs(image_list)
        return vec_obs

    def get_image(self, mask="handle") :
        self.get_image_async(mask)
        return self.get_image_wait()

    def get_observation_async(self, gt) :
        for remote in self.remotes:
            remote.send(("get_observation", gt))
        self.waiting = True

    def get_observation_wait(self) :
        observation_list = [remote.recv() for remote in self.remotes]
        self.waiting = False
        vec_obs = merge_obs(observation_list)
        return vec_obs

    def get_observation(self, gt=False) :
        self.get_observation_async(gt=gt)
        return self.get_observation_wait()

    def hand_move_to_async(
            self,
            poses,
            times,
            waits,
            planners,
            robot_frame,
            skip_moves,
            no_collision_with_front
        ) :
        if len(poses.shape) != 2:
            poses = [poses] * self.num_envs
        if not isinstance(times, list) :
            times = [times] * self.num_envs
        if not isinstance(waits, list) :
            waits = [waits] * self.num_envs
        if not isinstance(planners, list) :
            planners = [planners] * self.num_envs
        if not isinstance(robot_frame, list) :
            robot_frame = [robot_frame] * self.num_envs
        if not isinstance(skip_moves, list) :
            skip_moves = [skip_moves] * self.num_envs
        if not isinstance(no_collision_with_front, list) :
            no_collision_with_front = [no_collision_with_front] * self.num_envs
        for remote, p, s, w, pln, r, sk, n in zip(
                self.remotes,
                poses,
                times,
                waits,
                planners,
                robot_frame,
                skip_moves,
                no_collision_with_front
            ):
            remote.send(("hand_move_to", (p, s, w, pln, r, sk, n)))
        self.waiting = True

    def hand_move_to_wait(self) :
        ret_list = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return merge_obs(ret_list)

    def hand_move_to(
            self,
            pose,
            time=2,
            wait=1,
            planner="ik",
            robot_frame=False,
            skip_move=False,
            no_collision_with_front=True
        ) :
        self.hand_move_to_async(pose, time, wait, planner, robot_frame, skip_move, no_collision_with_front)
        return self.hand_move_to_wait()
    
    def cam_move_to_async(
            self,
            poses,
            times,
            waits,
            planners,
            robot_frame,
            skip_moves,
            no_collision_with_front
        ) :
        if isinstance(poses, sapien.Pose) or len(poses.shape) != 2:
            poses = [poses] * self.num_envs
        if not isinstance(times, list) :
            times = [times] * self.num_envs
        if not isinstance(waits, list) :
            waits = [waits] * self.num_envs
        if not isinstance(planners, list) :
            planners = [planners] * self.num_envs
        if not isinstance(robot_frame, list) :
            robot_frame = [robot_frame] * self.num_envs
        if not isinstance(skip_moves, list) :
            skip_moves = [skip_moves] * self.num_envs
        if not isinstance(no_collision_with_front, list) :
            no_collision_with_front = [no_collision_with_front] * self.num_envs
        for remote, p, s, w, pln, r, sk, n in zip(
                self.remotes,
                poses,
                times,
                waits,
                planners,
                robot_frame,
                skip_moves,
                no_collision_with_front
            ):
            remote.send(("cam_move_to", (p, s, w, pln, r, sk, n)))
        self.waiting = True

    def cam_move_to_wait(self) :
        ret_list = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return merge_obs(ret_list)

    def cam_move_to(
            self,
            pose,
            time=2,
            wait=1,
            planner="ik",
            robot_frame=False,
            skip_move=False,
            no_collision_with_front=True
        ) :
        self.cam_move_to_async(pose, time, wait, planner, robot_frame, skip_move, no_collision_with_front)
        return self.cam_move_to_wait()

    def gripper_move_to_async(
            self,
            poses,
            times,
            waits,
            planners,
            robot_frame,
            skip_moves,
            no_collision_with_front
        ) :
        if isinstance(poses, sapien.Pose) or len(poses.shape) != 2:
            poses = [poses] * self.num_envs
        if not isinstance(times, list) :
            times = [times] * self.num_envs
        if not isinstance(waits, list) :
            waits = [waits] * self.num_envs
        if not isinstance(planners, list) :
            planners = [planners] * self.num_envs
        if not isinstance(robot_frame, list) :
            robot_frame = [robot_frame] * self.num_envs
        if not isinstance(skip_moves, list) :
            skip_moves = [skip_moves] * self.num_envs
        if not isinstance(no_collision_with_front, list) :
            no_collision_with_front = [no_collision_with_front] * self.num_envs
        for remote, p, s, w, pln, r, sk, n in zip(
                self.remotes,
                poses,
                times,
                waits,
                planners,
                robot_frame,
                skip_moves,
                no_collision_with_front
            ):
            # remote.send(("gripper_move_to", (p, s, w, pln, r, sk, n)))
            remote.send(("gripper_move_to_1", (p, s, w, pln, r, sk, n)))
        self.waiting = True

    def gripper_move_to_wait(self) :
        ret_list = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return merge_obs(ret_list, with_obs = True)

    def gripper_move_to(
            self,
            pose,
            time=2,
            wait=1,
            planner="ik",
            robot_frame=False,
            skip_move=False,
            no_collision_with_front=True
        ) :
        self.gripper_move_to_async(pose, time, wait, planner, robot_frame, skip_move, no_collision_with_front)
        return self.gripper_move_to_wait()
    
    def robot_qpos(self) : 
        for remote in self.remotes:
            remote.send(("rbot_qpos", None))
        self.waiting = True
        ret_list = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return merge_obs(ret_list)

    def gripper_pose(self, robot_frame=False) :
        for remote in self.remotes:
            remote.send(("gripper_pose", robot_frame))
        self.waiting = True
        ret_list = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return merge_obs(ret_list)

    def camera_pose(self, robot_frame=False) :
        for remote in self.remotes:
            remote.send(("camera_pose", robot_frame))
        self.waiting = True
        ret_list = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return merge_obs(ret_list)

    def hand_pose(self, robot_frame=False) :
        for remote in self.remotes:
            remote.send(("hand_pose", robot_frame))
        self.waiting = True
        ret_list = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return merge_obs(ret_list)

    def robot_pose(self) :
        for remote in self.remotes:
            remote.send(("robot_pose", None))
        self.waiting = True
        ret_list = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return merge_obs(ret_list)
    
    def load(self, cfg):

        for remote in self.remotes :
            remote.send(("load", cfg))
        self.waiting = True
        for remote in self.remotes :
            remote.recv()
        self.waiting = False
        return None

    def class_method(
        self,
        method_name: str,
        *method_args,
        indices=None,
        **method_kwargs,
    ) -> List:
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("class_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def get_attr(
        self,
        name: str,
        indices=None
    ) -> List:
        """Get value of an attribute of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", name))
        return [remote.recv() for remote in target_remotes]

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def save_obs_list(self):

        for remote in self.remotes :
            remote.send(("save_obs_list", None))
        self.waiting = True

        for remote in self.remotes :
            remote.recv()
        self.waiting = False

        return None

    def draw_point_async(self, xyzs, colors) :
        for remote, xzy, color in zip(self.remotes, xyzs, colors):
            remote.send(("draw_point", (xzy, color)))
        self.waiting = True

    def draw_point_wait(self) :
        image_list = [remote.recv() for remote in self.remotes]
        self.waiting = False
        vec_obs = merge_obs(image_list)
        return vec_obs

    def draw_point(self, xyzs, colors = [1, 0, 0]) :
        if not isinstance(xyzs, list):
            xyzs = [xyzs] * self.num_envs
        if not isinstance(colors, list):
            colors = [colors] * self.num_envs

        self.draw_point_async(xyzs, colors)
        return self.draw_point_wait()

    def get_skip_list_async(self, batch_episode_dof) :
        for remote, episode_dof in zip(self.remotes, batch_episode_dof):
            remote.send(("determine_skip", episode_dof))
        self.waiting = True

    def get_skip_list_wait(self) :
        skip_list = [remote.recv() for remote in self.remotes]
        self.waiting = False

        return skip_list

    def get_skip_list(self, batch_episode_dof) :
        if not isinstance(batch_episode_dof, Union[list, np.ndarray]):
            batch_episode_dof = [batch_episode_dof] * self.num_envs

        self.get_skip_list_async(batch_episode_dof)
        return self.get_skip_list_wait()

    def predict_axis_async(
        self, 
        skip_list, 
    ):
        for remote, skip in zip(self.remotes, skip_list):
            remote.send(("predict_axis", skip))
        self.waiting = True

    def predict_axis_wait(self) :
        image_list = [remote.recv() for remote in self.remotes]
        self.waiting = False
        vec_obs = merge_obs(image_list)
        return vec_obs

    def predict_axis(
        self, 
        skip_list, 
    ):
        self.predict_axis_async(
            skip_list, 
        )
        return self.predict_axis_wait()

    def clear_point_async(self) :
        for remote in self.remotes:
            remote.send(("clear_point", None))
        self.waiting = True

    def clear_point_wait(self) :
        image_list = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return None

    def clear_point(self) :
        self.clear_point_async()
        return self.clear_point_wait()

    def save_axis_async(
        self, 
        batch_pivot_point, 
        batch_axis_direction, 
        batch_axis_color
    ):
        for remote, pivot_point, axis_direction, axis_color \
            in zip(self.remotes, batch_pivot_point, batch_axis_direction, batch_axis_color):
            remote.send(("save_axis", (pivot_point, axis_direction, axis_color)))
        self.waiting = True

    def save_axis_wait(self) :
        image_list = [remote.recv() for remote in self.remotes]
        self.waiting = False

        return None

    def save_axis(
        self, 
        batch_pivot_point, 
        batch_axis_direction, 
        batch_axis_color
    ):
        self.save_axis_async(
            batch_pivot_point, 
            batch_axis_direction, 
            batch_axis_color
        )
        return self.save_axis_wait()

    def draw_line_async(
        self, 
        batch_st_point, batch_ed_point, 
        batch_color
    ):
        for remote, st_point, ed_point, color \
            in zip(self.remotes, batch_st_point, batch_ed_point, batch_color):
            remote.send(("draw_line", (st_point, ed_point, color)))
        self.waiting = True

    def draw_line_wait(self) :
        image_list = [remote.recv() for remote in self.remotes]
        self.waiting = False
        vec_obs = merge_obs(image_list)
        return vec_obs

    def draw_line(
        self, 
        batch_st_point, batch_ed_point, 
        batch_color
    ):
        if not isinstance(batch_st_point, list):
            batch_st_point = [batch_st_point] * self.num_envs
        if not isinstance(batch_ed_point, list):
            batch_ed_point = [batch_ed_point] * self.num_envs
        if not isinstance(batch_color, list):
            batch_color = [batch_color] * self.num_envs

        self.draw_line_async(
            batch_st_point, batch_ed_point, 
            batch_color
        )
        return self.draw_line_wait()




