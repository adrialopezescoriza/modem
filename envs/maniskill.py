import gymnasium as gym
import numpy as np
import torch
from envs.utils import convert_observation_to_space

from mani_skill.utils.common import flatten_state_dict

from collections import deque

import mani_skill.envs
import envs.tasks.maniskill_stages


MANISKILL_TASKS = {
	'lift-cube': dict(
		env='LiftCube-v1',
		control_mode='pd_ee_delta_pose',
	),
	'pick-cube': dict(
		env='PickCube-v1',
		control_mode='pd_ee_delta_pose',
	),
	'pick-ycb': dict(
		env='PickSingleYCB-v1',
		control_mode='pd_ee_delta_pose',
	),
	'turn-faucet': dict(
		env='TurnFaucet-v1',
		control_mode='pd_ee_delta_pose',
	),
	'pick-place': dict(
		env='PickAndPlace_DrS_learn',
		control_mode='pd_ee_delta_pose',
		reward_mode='dense',
	),
	'stack-cube': dict (
		env='StackCube_DrS_learn',
		control_mode='pd_ee_delta_pose',
		reward_mode='dense', 
	),
	'peg-insertion': dict(
		env='PegInsertionSide_DrS_learn',
		control_mode='pd_ee_delta_pose',
		reward_mode='dense',
	),
	'two-robot-pick-cube': dict(
		env='TwoRobotPickCube_DrS_learn',
		control_mode='pd_ee_delta_pose',
		reward_mode='dense',
	),
	'two-robot-stack-cube': dict(
		env='TwoRobotStackCube_DrS_learn',
		control_mode='pd_ee_delta_pose',
		reward_mode='dense',
	),
	'lift-peg-upright': dict(
		env='LiftPegUpright_DrS_learn',
		control_mode='pd_ee_delta_pose',
		reward_mode='dense',
	),
	'poke-cube': dict(
		env='PokeCube_DrS_learn',
		control_mode='pd_ee_delta_pose',
		reward_mode='dense',
	),
	'humanoid-place-apple': dict(
		env='HumanoidPlaceApple_DrS_learn',
		control_mode='pd_joint_delta_pos',
		reward_mode='dense',
	),
	'humanoid-transport-box': dict(
		env='HumanoidTransportBox_DrS_learn',
		control_mode='pd_joint_delta_pos',
		reward_mode='dense',
	),
	## Semi-sparse reward tasks with stage-indicators
	'pick-place-semi': dict (
		env='PickAndPlace_DrS_learn',
		control_mode='pd_ee_delta_pose',
		reward_mode='semi_sparse', 
	),
	'stack-cube-semi': dict (
		env='StackCube_DrS_learn',
		control_mode='pd_ee_delta_pose',
		reward_mode='semi_sparse', 
	),
	'peg-insertion-semi': dict (
		env='PegInsertionSide_DrS_learn',
		control_mode='pd_ee_delta_pose',
		reward_mode='semi_sparse', 
	),
	'lift-peg-upright-semi': dict(
		env='LiftPegUpright_DrS_learn',
		control_mode='pd_ee_delta_pose',
		reward_mode='semi_sparse',
	),
	'poke-cube-semi': dict(
		env='PokeCube_DrS_learn',
		control_mode='pd_ee_delta_pose',
		reward_mode='semi_sparse', 
	),
	'two-robot-pick-cube-semi': dict(
		env='TwoRobotPickCube_DrS_learn',
		control_mode='pd_ee_delta_pose',
		reward_mode='semi_sparse',
	),
	'two-robot-stack-cube-semi': dict(
		env='TwoRobotStackCube_DrS_learn',
		control_mode='pd_ee_delta_pose',
		reward_mode='semi_sparse',
	),
	'humanoid-place-apple-semi': dict(
		env='HumanoidPlaceApple_DrS_learn',
		control_mode='pd_joint_delta_pos',
		reward_mode='semi_sparse',
	),
	'humanoid-transport-box-semi': dict(
		env='HumanoidTransportBox_DrS_learn',
		control_mode='pd_joint_delta_pos',
		reward_mode='semi_sparse',
	),
}

def select_obs(obs):
	"""
	Processes observations on the first nested level of the obs dictionary

	Args:
		keys: The keys
		obs: An array or dictionary of more nested observations or observation spaces 
	"""
	if not isinstance(obs, dict):
		return obs

	if "hand_camera" in obs["sensor_data"].keys():
		second_camera = "hand_camera"
	elif "ext_camera" in obs["sensor_data"].keys():
		second_camera = "ext_camera"
	elif "head_camera" in obs["sensor_data"].keys():
		second_camera = "head_camera"
	else:
		raise NotImplementedError
	image = torch.stack((obs['sensor_data']['base_camera']['rgb'].permute(0,3,1,2), obs['sensor_data'][second_camera]['rgb'].permute(0,3,1,2)), dim=1).squeeze()
	state_agent = flatten_state_dict(obs["agent"], use_torch=True)
	state_extra = flatten_state_dict(obs["extra"], use_torch=True)
	state = torch.cat([state_agent, state_extra], dim=-1).squeeze()
	return image, state

class ManiSkillWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		self.action_space = env.single_action_space
		self.max_episode_steps = cfg.max_episode_steps

		self._num_frames = cfg.get("frame_stack", 1)
		self._frames = deque([], maxlen=self._num_frames)
		self._state_frames = deque([], maxlen=self._num_frames)

		if hasattr(self.env.observation_space, 'spaces'):
			self.observation_space = convert_observation_to_space(self.get_obs())
		else:
			self.observation_space = gym.spaces.Box(
				low=np.full(
					env.single_observation_space.shape,
					-np.inf,
					dtype=np.float32),
				high=np.full(
					env.single_observation_space.shape,
					np.inf,
					dtype=np.float32),
				dtype=np.float32,
			)

	@property
	def state(self):
		return torch.cat(list(self._state_frames), dim=0)
	
	def _stacked_obs(self):
		assert len(self._frames) == self._num_frames
		return torch.cat(list(self._frames), dim=0)
	
	def rand_act(self):
		return torch.tensor(
			[self.action_space.sample().astype(np.float32) for _ in range(self.num_envs)],
			dtype=torch.float32, device=self.env.device)

	def reset(self, seed=None, **kwargs):
		self._t = 0
		obs, info = self.env.reset(seed=seed, **kwargs)
		for _ in range(self._num_frames):
			obs, state = select_obs(obs)
			self._frames.append(obs)
			self._state_frames.append(state)
		return self._stacked_obs()
	
	def step(self, action):
		for _ in range(self.cfg.action_repeat):
			obs, r, terminated, _, info = self.env.step(action)
			reward = r # Options: max, sum, min
		if isinstance(obs, dict):
			obs, state = select_obs(obs)
		self._frames.append(obs)
		self._state_frames.append(state)
		self._t += 1
		done = torch.tensor([self._t >= self.max_episode_steps] * self.num_envs)
		return self._stacked_obs(), reward, terminated, done, info
	
	def reward(self, **kwargs):
		return self.env.get_reward(obs=self.env.get_obs(), action=self.env.action_space.sample(), info=self.env.get_info())
	
	def get_obs(self, *args, **kwargs):
		obs, state = select_obs(self.env.get_obs())
		return obs

	@property
	def unwrapped(self):
		return self.env.unwrapped
	
	def render(self, *args, **kwargs):
		if kwargs.get("render_all", False):
			return self.env.render()
		return self.env.render()[0].cpu().numpy()


def make_env(cfg):
	"""
	Make ManiSkill2 environment.
	"""
	if cfg.task not in MANISKILL_TASKS:
		raise ValueError('Unknown task:', cfg.task)
	task_cfg = MANISKILL_TASKS[cfg.task]
	camera_resolution = dict(width=cfg.camera.get("image_size", 64), height=cfg.camera.get("image_size", 64))

	# WARNING: If one env is already in GPU, the other ones must also be in GPU
	env = gym.make(
		task_cfg['env'],
		obs_mode=cfg.obs,
		control_mode=task_cfg['control_mode'],
		num_envs=cfg.num_envs,
		reward_mode=task_cfg.get("reward_mode", None),
		render_mode='rgb_array',
		sensor_configs=camera_resolution,
		human_render_camera_configs=dict(width=384, height=384),
		reconfiguration_freq=1 if cfg.num_envs > 1 else None,
		sim_backend=cfg.get("sim_backend", "auto"),
		render_backend="auto",
	)
	env = ManiSkillWrapper(env, cfg)
	cfg.state_dim = select_obs(env.unwrapped.get_obs())[1].shape[-1]
	cfg.img_size = cfg.camera.image_size
	return env
