import numpy as np
import gymnasium as gym
import torch
import robosuite as suite

from envs.tasks.robosuite_stages import RobosuiteTask
from gymnasium.wrappers.rescale_action import RescaleAction

from envs.utils import convert_observation_to_space

class RobosuiteWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		self.observation_space = convert_observation_to_space(self.select_obs(self.get_observation()))

	def select_obs(self, obs):
		return np.stack([v for v in obs.values()])
	
	def rand_act(self):
		return self.action_space.sample().astype(np.float32)
	
	@property
	def state(self):
		" No state info in this environment"
		return torch.tensor([0])

	def reset(self, **kwargs):
		self._t = 0
		obs, info = super().reset(**kwargs)
		return self.select_obs(obs)

	def step(self, action):
		reward = 0
		action = action.numpy() if isinstance(action, torch.Tensor) else action
		for _ in range(self.cfg.action_repeat):
			obs, r, terminated, _, info = self.env.step(action)
			reward = r # Options: max, sum, min
		self._t += 1
		done = self._t >= self.max_episode_steps
		return self.select_obs(obs), reward, False, done, info

	def render(self, *args, **kwargs):
		return self.env.render(*args, **kwargs)
	
	def get_obs(self, *args, **kwargs):
		return self.select_obs(self.get_observation())

def make_env(cfg):
	"""
	Make Robosuite environment.
	"""
	parts = cfg.task.split("-") # Format is "env-task_id-reward_type"
	env_id = '-'.join(parts[1:-1])
	reward_mode = parts[-1]
	if not cfg.task.startswith('robosuite-'):
		raise ValueError('Unknown task:', cfg.task)
	env = RobosuiteTask(
			env_name=env_id,
			reward_type=reward_mode,
			cameras=(0, 1),
		    height=cfg.camera.image_size,
			width=cfg.camera.image_size,
            channels_first=True,
			control=None,
			set_done_at_success=False,
		)
	env = RescaleAction(env, -1.0, 1.0)
	env = RobosuiteWrapper(env, cfg)
	cfg.state_dim = 1
	cfg.episode_length = env.max_episode_steps
	cfg.img_size = cfg.camera.image_size
	return env