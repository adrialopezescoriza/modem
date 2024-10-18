import numpy as np
import gymnasium as gym
import torch

from collections import deque

from envs.tasks.bigym_stages import SUPPORTED_TASKS
from gymnasium.wrappers.rescale_action import RescaleAction

from envs.utils import convert_observation_to_space

class BiGymWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		self.observation_space = convert_observation_to_space(self.get_obs())

		self._num_frames = cfg.get("frame_stack", 1)
		self._frames = deque([], maxlen=self._num_frames)
		self._state_frames = deque([], maxlen=self._num_frames)

	def select_obs(self, obs):
		if self.cfg.obs == "state":
			return np.concatenate([v for v in obs.values()])
		state = np.empty((0,))
		images = []
		for k, v in obs.items():
			if k.startswith("proprioception"):
				state = np.concatenate((state, v))
			elif k.startswith("rgb"):
				images.append(v)
			else:
				raise NotImplementedError
		return np.stack(images), state

	def get_state(self):
		return self.select_obs(self.env.get_observation())[1]
	
	@property
	def state(self):
		return np.concatenate(list(self._state_frames), axis=0)
	
	def _stacked_obs(self):
		assert len(self._frames) == self._num_frames
		return np.concatenate(list(self._frames), axis=1)
	
	def rand_act(self):
		return self.action_space.sample().astype(np.float32)

	def reset(self, seed=None):
		self._t = 0
		obs, info = self.env.reset(seed=seed, options=None)
		for _ in range(self._num_frames):
			obs, state = self.select_obs(obs)
			self._frames.append(obs)
			self._state_frames.append(state)
		return self._stacked_obs()

	def step(self, action):
		reward = 0
		action = np.clip(action, self.action_space.low, self.action_space.high)
		action = action.numpy() if isinstance(action, torch.Tensor) else action
		for _ in range(self.cfg.action_repeat):
			obs, r, terminated, _, info = self.env.step(action)
			reward = r # Options: max, sum, min
		if isinstance(obs, dict):
			obs, state = self.select_obs(obs)
		self._frames.append(obs)
		self._state_frames.append(state)
		self._t += 1
		done = self._t >= self.max_episode_steps
		info["success"] = info["task_success"]
		return self._stacked_obs(), reward, False, done, info

	@property
	def unwrapped(self):
		return self.env.unwrapped

	def render(self, *args, **kwargs):
		return self.env.render(*args, **kwargs)
	
	def get_obs(self, *args, **kwargs):
		return self.select_obs(self.env.get_observation())[0]

def make_env(cfg):
	"""
	Make Meta-World environment.
	"""
	env_id = cfg.task.split("-",1)[-1]
	if not cfg.task.startswith('bigym-') or env_id not in SUPPORTED_TASKS:
		raise ValueError('Unknown task:', cfg.task)
	env = SUPPORTED_TASKS[env_id](
			obs_mode=cfg.obs, 
		    img_size=cfg.camera.image_size,
            render_mode=cfg.render_mode,
			start_seed=cfg.seed, 
		)
	cfg.obs = cfg.obs
	env = RescaleAction(env, -1.0, 1.0)
	env = BiGymWrapper(env, cfg)
	cfg.state_dim = env.get_state().shape[-1]
	cfg.img_size = cfg.camera.image_size
	cfg.episode_length = env.max_episode_steps
	return env