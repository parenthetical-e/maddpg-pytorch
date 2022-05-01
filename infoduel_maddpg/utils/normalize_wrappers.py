from copy import deepcopy
from statistics import mean

import numpy as np
import gym


def clip_reward(rewards, min_reward, max_reward):
    """Clip rewards"""
    return np.clip(rewards, min_reward, max_reward)


def fold_change(rewards, reference_reward, bias=0):
    """Estimate fold-change, with an optional bias."""
    #
    # Mostly this bias is there to allow our scheme to work when there
    # is NO variance in reward. This happens, as an example. in the "CartPole-v1" task.
    #
    # If we do not add a bias in cases like this then the reward
    # will be at zero always, and learning becomes impossible. Technically,
    # this bias breaks the notion of Fold-change. I need to handle
    # the corner case of no reward variance SOMEHOW. This bias is the
    # answer I decided on. Feel free to choose another, dear reader.
    rewards = (rewards - reference_reward) / reference_reward

    # If the ref is negative, we need to flip the signs
    # after normalization.
    rewards = rewards * np.sign(reference_reward)

    return rewards + bias


class RandomBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = int(buffer_size)
        self.buffer = list()

    def __len__(self):
        return len(self.buffer)

    def append(self, x):
        # Sanity
        assert len(self.buffer) <= self.buffer_size, "buffer too big"

        # Force min d and into np
        x = np.atleast_1d(x)

        # If smaller than buffer size, append
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(deepcopy(x))

        # Otherwise make a random insert
        else:
            i = np.random.randint(0, self.buffer_size)
            self.buffer[i] = deepcopy(x)

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
        buffer_size=1000,
    ):
        super().__init__(env)
        self.env = env

        # Init
        self.bias_reward = bias_reward
        # self.min_reference_reward = min_reference_reward
        self.intial_reference_reward = intial_reference_reward
        self.buffer_size = buffer_size

        # Construct independent buffers for all agents
        # and set initial ref
        self.buffer = {}
        self.reference_reward = {}
        for a in self.possible_agents:
            # Init
            self.buffer[a] = RandomBuffer(self.buffer_size)
            # Fill
            for _ in range(self.buffer_size):
                self.buffer[a].append(intial_reference_reward)
            # Update ref
            mean_reward = np.mean(self.buffer[a].sample())
            self.reference_reward[a] = mean_reward

        # Sanity
        assert len(self.reference_reward) == len(
            self.env.possible_agents
        ), "agent <> ref mismatch"

    def step(self, actions):
        next_states, rewards, dones, infos = self.env.step(actions)
        # Update reward memmory
        for a in rewards.keys():
            self.buffer[a].append(rewards[a])
        # Update reference rewardss
        for a in rewards.keys():
            mean_reward = np.mean(self.buffer[a].sample())
            self.reference_reward[a] = mean_reward
        # Est. fold-change
        rewards = self.normalize(rewards)

        return next_states, rewards, dones, infos

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
