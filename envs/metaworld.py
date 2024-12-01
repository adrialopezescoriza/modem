import numpy as np
import gymnasium as gym

from collections import deque

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from gymnasium.wrappers import TimeLimit
from envs.tasks.mw_stages import getRewardWrapper

import torchrl

class MetaWorldWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self._num_frames = cfg.get("frame_stack", 1)
        self._frames = deque([], maxlen=self._num_frames)
        self.env.mujoco_renderer.model.vis.global_.offwidth = cfg.camera.image_size
        self.env.mujoco_renderer.model.vis.global_.offheight = cfg.camera.image_size
        self.max_episode_steps = cfg.max_episode_steps

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=self._get_pixel_obs().shape,
            dtype=np.uint8,
        )
        self.action_space = self.env.action_space
        self.camera_name = "corner2"
        self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]

    @property
    def state(self):
        state = self._state_obs.astype(np.float32)
        return np.concatenate((state[:4], state[18 : 18 + 4])) if self._num_frames >= 2 else state[:4]

    def _get_pixel_obs(self):
        return np.expand_dims(self.render().transpose(2, 0, 1), axis=0)

    def _stacked_obs(self):
        assert len(self._frames) == self._num_frames
        return np.concatenate(list(self._frames), axis=0)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs = self.env.step(np.zeros_like(self.env.action_space.sample()))[0].astype(
            np.float32
        )
        self._state_obs = obs
        obs = self._get_pixel_obs()
        for _ in range(self._num_frames):
            self._frames.append(obs)
        return self._stacked_obs()

    def step(self, action):
        reward = 0
        for _ in range(self.cfg.action_repeat):
            obs, r, _, _, info = self.env.step(action)
            reward = r
        obs = obs.astype(np.float32)
        self._state_obs = obs
        obs = self._get_pixel_obs()
        self._frames.append(obs)
        return self._stacked_obs(), reward, False, False, info

    def render(self, *args, **kwargs):
        if self.env.camera_name in ("corner", "corner2"):
            return np.flip(self.env.render(*args, **kwargs), axis=0)
        return self.env.render(*args, **kwargs)

def make_env(cfg):
	"""
	Make Meta-World environment.
	"""
	parts = cfg.task.split("-") # Format is "env-task_id-reward_type"
	env_id = '-'.join(parts[1:-1])  + "-v2-goal-observable"
	if not cfg.task.startswith('mw-') or env_id not in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE:
		raise ValueError('Unknown task:', cfg.task)
	cfg.reward_mode = parts[-1]
	cfg.state_dim = 4 * cfg.get("frame_stack", 1)
	if cfg.reward_mode == "semi":
		cfg.reward_mode = "semi_sparse"
	env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_id](
			seed=cfg.seed, 
			render_mode=cfg.render_mode,
		)
	env.camera_name = "corner2"
	env._freeze_rand_vec = False
	env = getRewardWrapper(env_id)(env, cfg)
	env = MetaWorldWrapper(env, cfg)
	env = TimeLimit(env, max_episode_steps=cfg.episode_length)
	return env