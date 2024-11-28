from copy import deepcopy
import warnings

from collections import defaultdict

import gymnasium as gym
import numpy as np
import torch
import random

def missing_dependencies(task):
	raise ValueError(f'Missing dependencies for task {task}; install dependencies to use this environment.')

try:
	from envs.maniskill import make_env as make_maniskill_env
except:
	make_maniskill_env = missing_dependencies
try:
	from envs.metaworld import make_env as make_metaworld_env
except:
	make_metaworld_env = missing_dependencies
try:
	from envs.bigym import make_env as make_bigym_env
except:
	make_bigym_env = missing_dependencies
try:
	from envs.robosuite import make_env as make_robosuite_env
except:
	make_robosuite_env = missing_dependencies


warnings.filterwarnings('ignore', category=DeprecationWarning)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class DefaultDictWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.env = env

    @property
    def unwrapped(self):
        return self.env.unwrapped
    def step(self, action):
        obs, reward, _, done, info = self.env.step(action)
        info = defaultdict(float, info)
        return obs, reward, done, info

def make_env(cfg):
	"""
	Make a vectorized environment for TD-MPC2 experiments.
	"""
	gym.logger.set_level(40)
	env = None
	for fn in [make_maniskill_env, make_metaworld_env, make_bigym_env, make_robosuite_env]:
		try:
			env = fn(cfg)
			break
		except ValueError:
			pass
	if env is None:
		raise ValueError(f'Failed to make environment "{cfg.task}": please verify that dependencies are installed and that the task exists.')
	
	env = DefaultDictWrapper(env)
	cfg.obs_shape = tuple(int(x) for x in env.observation_space.shape)
	cfg.action_shape = tuple(int(x) for x in env.action_space.shape)
	cfg.action_dim = env.action_space.shape[0]
	return env
