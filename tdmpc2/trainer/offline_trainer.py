import os
from copy import deepcopy
from time import time
from pathlib import Path
from glob import glob
from tqdm import tqdm 

import numpy as np
import torch
from tqdm import tqdm

from common.buffer import Buffer
from trainer.base import Trainer
from tensordict import TensorDict

class OfflineTrainer(Trainer):
	"""Trainer class for multi-task offline TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._start_time = time()
	
	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		results = dict()
		for task_idx in tqdm(range(len(self.cfg.tasks)), desc='Evaluating'):
			ep_rewards, ep_successes = [], []
			for _ in range(self.cfg.eval_episodes):
				obs, done, ep_reward, t = self.env.reset(task_idx), False, 0, 0
				while not done:
					torch.compiler.cudagraph_mark_step_begin()
					action = self.agent.act(obs, t0=t==0, eval_mode=True, task=task_idx)
					obs, reward, done, info = self.env.step(action)
					ep_reward += reward
					t += 1
				ep_rewards.append(ep_reward)
				ep_successes.append(info['success'])
			results.update({
				f'episode_reward+{self.cfg.tasks[task_idx]}': np.nanmean(ep_rewards),
				f'episode_success+{self.cfg.tasks[task_idx]}': np.nanmean(ep_successes),})
		return results
	
	def _load_dataset(self):
		"""Load dataset for offline training."""
		loaded = np.load( self.cfg.data_dir, allow_pickle = True )
		obs, actions, rewards, lens, task_name = loaded['obs'].item(), loaded['action'], loaded['reward'], loaded['traj_lengths'], loaded['task'].item()
		
		obs = np.concatenate([v for k,v in obs.items()], axis = -1) # combine obs dictionary into a state vector
		ep_ends = list( np.cumsum(lens) )
		ep_starts = [0] + ep_ends[:-1]
		# Create buffer for sampling
		_cfg = deepcopy(self.cfg)
		_cfg.episode_length = 300
		_cfg.buffer_size = 1000 * 300 # 1000 trajectories, maximally 300 steps
		_cfg.steps = _cfg.buffer_size
		self.buffer = Buffer(_cfg)
		
		for ep_start, ep_end in tqdm(zip(ep_starts, ep_ends), desc='Loading data'):
			_obs, _actions, _rewards = obs[ep_start:ep_end], actions[ep_start:ep_end], rewards[ep_start:ep_end]
			terminated, tasks = np.zeros_like(_rewards),  np.zeros_like(_rewards)
			terminated[-1] = True
			td =  TensorDict({
					'obs': torch.tensor(_obs).unsqueeze(0).float(),
					'action': torch.tensor(_actions) .unsqueeze(0).float(),
					'reward': torch.tensor(_rewards) .unsqueeze(0).float(),
					# 'terminated': torch.tensor(terminated) .unsqueeze(0),
					'task': torch.tensor(tasks).unsqueeze(0).float(), # a dummy value
					
				}, batch_size = [1, len(_obs)])
			
			self.buffer.load( td )
		
		expected_episodes = _cfg.buffer_size // _cfg.episode_length
		if self.buffer.num_eps != expected_episodes:
			print(f'WARNING: buffer has {self.buffer.num_eps} episodes, expected {expected_episodes} episodes for {self.cfg.task} task set.')
		
	def train(self):
		"""Train a TD-MPC2 agent."""
		
		self._load_dataset()
		
		print(f'Training agent for {self.cfg.steps} iterations...')
		metrics = {}
		for i in tqdm( range(self.cfg.steps) ):

			# Update agent
			train_metrics = self.agent.update(self.buffer)
			
			# Evaluate agent periodically
			if i % self.cfg.eval_freq == 0:
				metrics = {
					'iteration': i,
					'elapsed_time': time() - self._start_time,
				}
				metrics.update(train_metrics)
				
				self.logger.log(metrics, 'pretrain')
			
		self.logger.finish(self.agent)
