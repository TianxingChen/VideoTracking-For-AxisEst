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

try:
    import torch
except ImportError:
    raise ImportError("To use ManiSkill2 VecEnv, please install PyTorch first.")

import utils.logger as logger

def find_available_port():
    # https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
        server_address = f"localhost:{port}"
    return server_address


def _worker(
    rank: int,
    remote: Connection,
    parent_remote: Connection,
    env_fn: Callable,
):
    # NOTE(jigu): Set environment variables for ManiSkill2
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    parent_remote.close()

    try:
        env = env_fn()
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, done, info = env.step(data)
                remote.send((obs, reward, done, info))
            elif cmd == "get_state":
                state = env.get_state()
                remote.send(state)
            elif cmd == "reset":
                obs = env.reset()
                remote.send(obs)
            elif cmd == "close":
                remote.close()
                break
            elif cmd == "class_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == "handshake":
                remote.send(None)
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
class SingleVecEnv() :

    def __init__(self, env) :

        super().__init__()

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.state_space = env.state_space
        self.max_episode_length = 256
        self.exp_name = "RL test"
        self.num_envs = 1
        self.env = env
    
    def reset(self) :

        obs = self.env.reset()

        if "image" in obs :
            image_obs = obs.pop("image")
        
        obs = {
            "state": torch.from_numpy(flatten_dict(obs)[np.newaxis, ...]).float(),
            "image": numpy_dict_to_tensor_dict(image_obs)
        }

        return obs

    def step(self, action : torch.Tensor) :

        obs, reward, done, info = self.env.step(action[0].numpy())

        if "image" in obs :
            image_obs = obs.pop("image")

        obs = {
            "state": torch.from_numpy(flatten_dict(obs)[np.newaxis, ...]).float(),
            "image": numpy_dict_to_tensor_dict(image_obs)
        }

        return obs, reward*torch.ones((1,)).float(), done*torch.ones((1,)).float(), info

    def get_state(self) :

        obs = self.env.get_state()

        if "image" in obs :
            image_obs = obs.pop("image")

        obs = {
            "state": torch.from_numpy(flatten_dict(obs)[np.newaxis, ...]).float(),
            "image": numpy_dict_to_tensor_dict(image_obs)
        }

        return obs

    def class_method(self, method, *args, **kwargs) :

        return getattr(self.env, method)(*args, **kwargs)

class VecEnv:
    """Vectorized environment modified from Stable Baselines3 for ManiSkill2.
    Image observations can stay on GPU to avoid unnecessary data transfer.

    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    .. warning::
        The tensor observations are buffered. Make a copy to avoid from overwriting them.

    :param env_fns: Environments to run in subprocesses
    :param start_method: method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    :param server_address: The network address of the SAPIEN RenderServer.
        If "auto", the server will be created automatically at an avaiable port.
        Otherwise, it should be a networkd address, e.g. "localhost:12345".
    :param server_kwargs: keyword arguments for sapien.RenderServer
    """

    device: torch.device

    def __init__(
        self,
        env_fns: List[Callable],
        start_method: Optional[str] = None,
        server_address: str = "auto",
        server_kwargs: dict = None,
    ):
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
        args = (0, work_remote, remote, env_fns[0])
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
        # ---------------------------------------------------------------------------- #

        n_envs = len(env_fns)
        self.num_envs = n_envs

        # Allocate numpy buffers
        self.non_image_obs_space = deepcopy(self.observation_space)
        self.image_obs_space = self.non_image_obs_space.spaces.pop("image")
        self._last_obs_np = [None for _ in range(n_envs)]
        self._obs_np_buffer = create_np_buffer(self.non_image_obs_space, n=n_envs)

        self.non_image_state_space = deepcopy(self.state_space)
        self.image_state_space = self.non_image_state_space.spaces.pop("image")
        self._last_state_np = [None for _ in range(n_envs)]
        self._state_np_buffer = create_np_buffer(self.non_image_state_space, n=n_envs)

        # Start RenderServer
        if server_address == "auto":
            server_address = find_available_port()
        self.server_address = server_address
        server_kwargs = {} if server_kwargs is None else server_kwargs
        self.server = sapien.RenderServer(**server_kwargs)
        self.server.start(self.server_address)
        logger.log.info(f"RenderServer is running at: {server_address}")

        # Wrap env_fn
        for i, env_fn in enumerate(env_fns):
            client_kwargs = {"address": self.server_address, "process_index": i}
            env_fns[i] = partial(
                env_fn, renderer="client", renderer_kwargs=client_kwargs
            )

        # Initialize workers
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for rank in range(n_envs):
            work_remote = self.work_remotes[rank]
            remote = self.remotes[rank]
            env_fn = env_fns[rank]
            args = (rank, work_remote, remote, env_fn)
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

        # Infer texture names
        texture_names = set()
        for cam_space in self.image_obs_space.spaces.values():
            texture_names.update(cam_space.spaces.keys())
        self.texture_names = tuple(texture_names)

        # Allocate torch buffers
        # A list of [n_envs, n_cams, H, W, C] tensors
        if len(self.texture_names) :
            self._obs_torch_buffer: List[
                torch.Tensor
            ] = self.server.auto_allocate_torch_tensors(self.texture_names)
            self.device = self._obs_torch_buffer[0].device

    # ---------------------------------------------------------------------------- #
    # States
    # ---------------------------------------------------------------------------- #
    def _update_state_buffer(self, obs_list, indices=None):
        indices = self._get_indices(indices)
        for i, obs in zip(indices, obs_list):
            self._last_state_np[i] = obs
        return stack_obs(
            self._last_state_np, self.non_image_state_space, self._state_np_buffer
        )

    # ---------------------------------------------------------------------------- #
    # Observations
    # ---------------------------------------------------------------------------- #
    def _update_obs_buffer(self, obs_list, indices=None):
        indices = self._get_indices(indices)
        for i, obs in zip(indices, obs_list):
            self._last_obs_np[i] = obs
        return stack_obs(
            self._last_obs_np, self.non_image_obs_space, self._obs_np_buffer
        )

    @torch.no_grad()
    def _get_torch_observations(self):
        self.server.wait_all()

        if len(self.texture_names) == 0:
            return dict(image={})

        tensor_dict = {}
        for i, name in enumerate(self.texture_names):
            tensor_dict[name] = self._obs_torch_buffer[i]

        # NOTE(jigu): Efficiency might not be optimized when using more cameras
        image_obs = {}
        for cam_idx, cam_uid in enumerate(self.image_obs_space.spaces.keys()):
            image_obs[cam_uid] = {}
            cam_space = self.image_obs_space[cam_uid]
            for tex_name in cam_space:
                tensor = tensor_dict[tex_name][:, cam_idx]  # [B, H, W, C]
                if tensor.shape[1:3] != cam_space[tex_name].shape[0:2]:
                    h, w = cam_space[tex_name].shape[0:2]
                    tensor = tensor[:, :h, :w]
                image_obs[cam_uid][tex_name] = tensor
        
        return dict(image=image_obs)

    # ---------------------------------------------------------------------------- #
    # Interfaces
    # ---------------------------------------------------------------------------- #
    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        if seed is None:
            seed = np.random.randint(0, 2**32)
        for idx, remote in enumerate(self.remotes):
            remote.send(("class_method", ("seed", [seed + idx], {})))
        return [remote.recv() for remote in self.remotes]

    def reset_async(self, indices=None):
        remotes = self._get_target_remotes(indices)
        for remote in remotes:
            remote.send(("reset", None))
        self.waiting = True

    def reset_wait(self, indices=None):
        remotes = self._get_target_remotes(indices)
        results = [remote.recv() for remote in remotes]
        self.waiting = False
        vec_obs = self._get_torch_observations()
        self._update_obs_buffer(results, indices)
        vec_obs.update(deepcopy(self._obs_np_buffer))
        return vec_obs

    def reset(self, indices=None):
        self.reset_async(indices=indices)
        return self.reset_wait(indices=indices)

    def step_async(self, actions: np.ndarray) -> None:
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs_list, rews, dones, infos = zip(*results)
        vec_obs = self._get_torch_observations()
        self._update_obs_buffer(obs_list)
        vec_obs.update(deepcopy(self._obs_np_buffer))
        return vec_obs, torch.tensor(rews).float(), torch.tensor(dones).float(), infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()
    
    def state_async(self) :
        for remote in self.remotes:
            remote.send(("get_state", None))
        self.waiting = True
    
    def state_wait(self) :
        state_list = [remote.recv() for remote in self.remotes]
        self.waiting = False
        vec_state = self._get_torch_observations()
        self._update_state_buffer(state_list)
        vec_state.update(deepcopy(self._state_np_buffer))
        return vec_state
    
    def get_state(self) :
        self.state_async()
        return self.state_wait()

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

    def render(self, mode=""):
        raise NotImplementedError

    def get_attr(self, attr_name: str, indices=None) -> List:
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

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

    def env_is_wrapped(
        self, wrapper_class: Type[gym.Wrapper], indices=None
    ) -> List[bool]:
        """Check if environments are wrapped with a given wrapper."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("is_wrapped", wrapper_class))
        return [remote.recv() for remote in target_remotes]

    @property
    def unwrapped(self) -> "VecEnv":
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

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

    def __repr__(self):
        return "{}({})".format(
            self.__class__.__name__, self.class_method("__repr__", indices=0)[0]
        )


def stack_observation_space(space: spaces.Space, n: int):
    if isinstance(space, spaces.Dict):
        sub_spaces = [
            (key, stack_observation_space(subspace, n))
            for key, subspace in space.spaces.items()
        ]
        return spaces.Dict(sub_spaces)
    elif isinstance(space, spaces.Box):
        shape = (n,) + space.shape
        low = np.broadcast_to(space.low, shape)
        high = np.broadcast_to(space.high, shape)
        return spaces.Box(low=low, high=high, shape=shape, dtype=space.dtype)
    else:
        raise NotImplementedError(
            "Unsupported observation space: {}".format(type(space))
        )


def create_np_buffer(space: spaces.Space, n: int):
    if isinstance(space, spaces.Dict):
        return {
            key: create_np_buffer(subspace, n) for key, subspace in space.spaces.items()
        }
    elif isinstance(space, spaces.Box):
        return np.zeros((n,) + space.shape, dtype=space.dtype)
    else:
        raise NotImplementedError(
            "Unsupported observation space: {}".format(type(space))
        )

def stack_obs(obs: Sequence, space: spaces.Space, buffer: Optional[np.ndarray] = None):
    if isinstance(space, spaces.Dict):
        ret = {}
        for key in space:
            _obs = [o[key] for o in obs]
            _buffer = None if buffer is None else buffer[key]
            ret[key] = stack_obs(_obs, space[key], buffer=_buffer)
        return ret
    elif isinstance(space, spaces.Box):
        return np.stack(obs, out=buffer)
    else:
        raise NotImplementedError(type(space))

class VecEnvWrapper(VecEnv):
    def __init__(self, venv: VecEnv):
        self.venv = venv
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_space
        self.state_space = venv.state_space
        self.action_space = venv.action_space

    def seed(self, seed: Optional[int] = None):
        return self.venv.seed(seed)

    def reset_async(self, *args, **kwargs):
        self.venv.reset_async(*args, **kwargs)

    def reset_wait(self, *args, **kwargs):
        return self.venv.reset_wait(*args, **kwargs)

    def step_async(self, actions: np.ndarray):
        self.venv.step_async(actions)

    def step_wait(self):
        return self.venv.step_wait()

    def close(self):
        return self.venv.close()

    def render(self, mode=""):
        return self.venv.render(mode)

    def get_attr(self, attr_name: str, indices=None) -> List:
        return self.venv.get_attr(attr_name, indices)

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        return self.venv.set_attr(attr_name, value, indices)

    def class_method(
        self,
        method_name: str,
        *method_args,
        indices=None,
        **method_kwargs,
    ) -> List:
        return self.venv.class_method(
            method_name, *method_args, indices=indices, **method_kwargs
        )

    def env_is_wrapped(
        self, wrapper_class: Type[gym.Wrapper], indices=None
    ) -> List[bool]:
        return self.venv.env_is_wrapped(wrapper_class, indices)

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            return getattr(self.venv, name)


class VecEnvObservationWrapper(VecEnvWrapper):
    def reset_wait(self, **kwargs):
        observation = self.venv.reset_wait(**kwargs)
        return self.observation(observation)

    def step_wait(self):
        observation, reward, done, info = self.venv.step_wait()
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        raise NotImplementedError
