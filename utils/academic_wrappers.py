"""An set of 'AcademicEnv' gym wrappers.

AcademicEnvs add learning 'agents' to Gym environments. These 'agents' take no actions themselves. They only judge the value of actions taken by other external agents in order to generate 'intrinsic reward'.

In other words, 'If you want to know if intrinsic rewaards will help, just add an academic'.

"""

import gym
import numpy as np
from gym import spaces
from typing import Dict, List, Tuple, Type, Union

import torch
import torch as th
import torch.optim as optim
import torch.nn.functional as F
import random
import torch.optim as optim
import torchvision.transforms as T
import torch.nn as nn

from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.utils import get_device

from dualer.common import create_mlp
from dualer.common import FrozenConv
from dualer.common import FrozenTLP


class StatePrediction(gym.Wrapper):
    """An academic wrapper for doing state prediction."""

    def __init__(
        self,
        env,
        reward_weight=0.0,
        intrinsic_weight=1.0,
        reset_networks=False,
        mode="memory",  # optional
        lr=0.01,
        network_hidden=None,
        device="auto",
    ):
        # Init gym
        super().__init__(env)
        self.env = env

        self.intrinsic_weight = intrinsic_weight
        self.reward_weight = reward_weight
        self.reset_networks = reset_networks
        self.mode = mode
        self.device = get_device(device)

        # Init net
        self.lr = float(lr)
        self.network_hidden = network_hidden
        # TODO: fix for agents
        self.network_input = env.observation_space.shape[0]  # Assume state vec
        self.network_output = self.network_input  # Same size

        self._init_network()
        self._init_optimizer()

        self._state = self.env.reset()

    def _init_network(self):
        """Build one network pair per zoo (env) agent."""

        self.network = {}
        self.target = {}

        for i, a in enumerate(self.possible_agents):
            # One network / env
            self.network[a] = nn.Sequential(
                *create_mlp(
                    self.network_input[i],
                    self.network_output[i],
                    self.network_hidden,
                )
            )

            self.target[a] = nn.Sequential(
                *create_mlp(
                    self.network_input[i],
                    self.network_output[i],
                    self.network_hidden,
                )
            )

            # To device
            self.network[a].to(self.device)
            self.target[a].to(self.device)

    def _init_optimizer(self):
        self.optimizer = {}
        for a in range(self.possible_agents):
            self.optimizer[a] = optim.Adam(self.network[a].parameters(), self.lr)

    def _learn(self, state, next_state):
        intrinsics = []
        losses = []
        for i, a in enumerate(self.agents):
            network = self.network[a]  # brevity
            target = self.target[a]
            optimizer = self.optimizer[a]

            # We sync up the 'target' net here so
            # when we learn below we have another
            # memory to compare to
            target.load_state_dict(network.state_dict())
            target.to(self.device)

            # Match state to fmt networks/opt expect
            # aka torch tensors
            state_i = state[a]
            next_state_i = next_state[a]
            state_i = torch.tensor(state_i).float().to(self.device)
            next_state_i = torch.tensor(next_state_i).float().to(self.device)

            # Learn
            pred_state_i = network(state_i)
            loss = F.huber_loss(next_state_i, pred_state_i)
            optimizer.zero_grad()
            loss.backward()

            # Don't want HUGE grads, so we clip 'em
            # before we use 'em
            for param in network.parameters():
                param.grad.data.clamp_(-1, 1)
                optimizer.step()

            # Choose an intrinsic reward:
            # - Prediction error
            if self.mode == "prediction":
                intrinsic = loss.detach().cpu().squeeze().numpy()
            # - Info. value (Peterson & Verstynen)
            elif self.mode == "memory":
                with th.no_grad():
                    # Extract
                    old_memory = th.cat(
                        [p.flatten() for p in self.target[a].parameters()]
                    )
                    memory = th.cat([p.flatten() for p in self.network[a].parameters()])
                    # Scale
                    old_memory = F.normalize(old_memory, p=2, dim=0)
                    memory = F.normalize(memory, p=2, dim=0)
                    # Est value
                    intrinsic = th.norm(old_memory - memory, p=2).detach().cpu().numpy()
            else:
                raise ValueError("mode not known")

            intrinsics.append(intrinsic)
            losses.append(
                {"network_loss": loss.detach().cpu().squeeze().numpy().item()}
            )

        return intrinsics, losses

    def step(self, actions):
        next_states, rewards, dones, infos = self.env.step(actions)
        # Estimate the academic reward
        intrinsics, intrinsic_infos = self._learn(self._state, next_states)
        # Mix rewards
        totals = (rewards * self.reward_weight) + (intrinsics * self.intrinsic_weight)
        # Shift
        self._state = next_states.copy()
        # Update info
        for i, a in enumerate(self.agents):
            infos[a]["env_reward"] = rewards[a].item()
            infos[a]["intrinsic_reward"] = intrinsics[i].item()
            infos[a].update(intrinsic_infos[i])

        return next_states, totals, dones, infos

    def reset(self):
        state = self.env.reset()
        if self.reset_networks:
            self._init_network()
            self._init_optimizer()
        return state
