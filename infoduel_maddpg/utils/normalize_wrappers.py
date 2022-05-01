from collections import deque
from copy import deepcopy
from statistics import mean

import numpy as np
import gym


class RunningStats:
    """Computes running mean and standard deviation.

    Adapted from:
        - https://gist.github.com/fvisin/5a10066258e43cf6acfa0a474fcdb59f
        - http://mathcentral.uregina.ca/QQ/database/QQ.09.02/carlos1.html
    """

    def __init__(self):
        self.n = 0.0

    def clear(self):
        self.n = 0.0

    def append(self, x, per_dim=True):
        x = np.array(x).copy().astype("float16")
        # process input
        if per_dim:
            self._update_params(x)
        else:
            for el in x.flatten():
                self._update_params(el)

    def _update_params(self, x):
        self.n += 1
        if self.n == 1:
            self.m = x
            self.s = 0.0
        else:
            prev_m = self.m.copy()
            self.m += (x - self.m) / self.n
            self.s += (x - prev_m) * (x - self.m)

    def mean(self):
        return self.m if self.n else 0.0

    def var(self):
        return self.s / (self.n) if self.n else 0.0

    def std(self):
        return np.sqrt(self.variance())


class SequentialBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = int(buffer_size)
        self.buffer = deque([], self.buffer_size)

    def __len__(self):
        return len(self.buffer)

    def append(self, x):
        x = np.atleast_1d(x)
        self.buffer.append(x)

    def sample(self):
        return self.buffer


class ClipReward(gym.Wrapper):
    """Clip rewards between [min_reward, max_reward]"""

    def __init__(self, env, min_reward=0, max_reward=1):
        super().__init__(env)
        self.env = env
        self.min_reward = min_reward
        self.max_reward = max_reward

    def step(self, actions):
        next_states, rewards, dones, infos = self.env.step(actions)
        rewards = self.normalize(rewards)
        return next_states, rewards, dones, infos

    def normalize(self, rewards):
        clip_rewards = {}
        for a in rewards.keys():
            clip_rewards[a] = np.clip(rewards[a], self.min_reward, self.max_reward)
        return clip_rewards

    def reset(self):
        states = self.env.reset()
        return states


class MovingFoldChangeReward(gym.Wrapper):
    """Normalize rewards using Fold-change detection and a running
    mean reference reward.

    A biological/psychological motivated approach to reward normalization
    inspired by:

    - Adler, M., and Alon, U. (2018). Fold-change detection in biological
    systems. Current Opinion in Systems Biology 8, 81–89.
    - Karin, O., and Alon, U. (2021). The dopamine circuit as a reward-taxis navigation system. BioRxiv 439955, 30.
    """

    def __init__(
        self,
        env,
        intial_reference_reward=0.01,
        # min_reference_reward=0.001,
        bias_reward=0.0,
        # buffer_size=1000,
    ):
        super().__init__(env)
        self.env = env

        # Init
        self.bias_reward = bias_reward
        self.intial_reference_reward = intial_reference_reward

        # Construct independent buffers for all agents
        self.buffer = {}
        self.reference_reward = {}
        for a in self.possible_agents:
            self.buffer[a] = RunningStats()
            # ...and set initial value
            self.buffer[a].append(intial_reference_reward)
            self.reference_reward[a] = self.buffer[a].mean()

        # Sanity
        assert len(self.reference_reward) == len(
            self.env.possible_agents
        ), "agent <> ref mismatch"

    def step(self, actions):
        next_states, rewards, dones, infos = self.env.step(actions)
        self.update_reference(rewards)
        rewards = self.normalize(rewards)

        return next_states, rewards, dones, infos

    def update_reference(self, rewards) -> None:
        # Update reward memmory
        for a in rewards.keys():
            self.buffer[a].append(rewards[a])
        # Update reference rewardss
        for a in rewards.keys():
            # mean_reward = np.mean(self.buffer[a].sample())
            self.reference_reward[a] = self.buffer[a].mean()

    def normalize(self, rewards):
        fold_rewards = {}
        for a in rewards.keys():
            # Fold change is defined as, (x - ref) / ref
            delta = rewards[a] - self.reference_reward[a]
            fold_reward = delta / self.reference_reward[a]
            # Correct for sign differences. For example if ref is negative but
            # reward is positive
            fold_reward = fold_reward * np.sign(self.reference_reward[a])
            # Add a bias that useful and needed when there is no
            # variability in the rewards at all
            fold_reward += self.bias_reward
            # !
            fold_rewards[a] = deepcopy(fold_reward)

        return fold_rewards

    def reset(self):
        state = self.env.reset()
        return state
