import argparse

# import imp
import torch

# import time
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path

# from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

# from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from algorithms.maddpg import MADDPG

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from pettingzoo import mpe
from supersuit import stable_baselines3_vec_env_v0
from supersuit import gym_vec_env_v0
from supersuit import pettingzoo_env_to_vec_env_v1

USE_CUDA = False  # torch.cuda.is_available()


def make_env(env_id, n_rollout_threads, seed, discrete_action=False):
    # A clusre dance (for some reason to init)
    Env = getattr(mpe, env_id)

    def get_env_fn(rank):
        def init_env():
            if discrete_action:
                env = Env.parallel_env(continuous_actions=False)
            else:
                env = Env.parallel_env(continuous_actions=True)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)

            return env

        return init_env

    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])


def run(config):
    model_dir = Path("./models") / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = "run1"
    else:
        exst_run_nums = [
            int(str(folder.name).split("run")[1])
            for folder in model_dir.iterdir()
            if str(folder.name).startswith("run")
        ]
        if len(exst_run_nums) == 0:
            curr_run = "run1"
        else:
            curr_run = "run%i" % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / "logs"
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)
    Env = getattr(mpe, config.env_id)
    env = Env.parallel_env(continuous_actions=True)
    env.reset()
    # Make access to 'space' info ass expected through the existing codebase
    action_space = [env.action_space(a) for a in env.possible_agents]
    observation_space = [env.observation_space(a) for a in env.possible_agents]
    print("action_space", action_space)
    print("observation_space", observation_space)

    # observation_space = [env.observation_space(a) for a in env.possible_agents]
    # env = gym_vec_env_v0(env, config.n_rollout_threads)
    # env = make_env(
    #     config.env_id, config.n_rollout_threads, config.seed, config.discrete_action
    # )
    # env = pettingzoo_env_to_vec_env_v1(env)
    # env = stable_baselines3_vec_env_v0(env, config.n_rollout_threads)
    maddpg = MADDPG.init_from_env(
        env,
        agent_alg=config.agent_alg,
        adversary_alg=config.adversary_alg,
        tau=config.tau,
        lr=config.lr,
        hidden_dim=config.hidden_dim,
    )
    # maddpg = MADDPG()
    replay_buffer = ReplayBuffer(
        config.buffer_length,
        maddpg.nagents,
        [obsp.shape[0] for obsp in observation_space],
        [acsp.shape[0] if isinstance(acsp, Box) else acsp.n for acsp in action_space],
    )
    t = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print(
            "Episodes %i-%i of %i"
            % (ep_i + 1, ep_i + 1 + config.n_rollout_threads, config.n_episodes)
        )
        obs = env.reset()
        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
        maddpg.prep_rollouts(device="cpu")

        explr_pct_remaining = (
            max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        )
        maddpg.scale_noise(
            config.final_noise_scale
            + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining
        )
        maddpg.reset_noise()

        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            # print(obs)
            # torch_obs = [
            #     torch.tensor(np.vstack(obs[:, i]), requires_grad=False)
            #     for i in range(maddpg.nagents)
            # ]
            # torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
            #                       requires_grad=False)
            #              for i in range(maddpg.nagents)]
            print("----------------------------")
            print("ei_i", et_i)

            torch_obs = [
                torch.tensor(obs[a], requires_grad=False).unsqueeze(0)
                for a in env.possible_agents
            ]

            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            print("torch_agent_actions:", torch_agent_actions)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            # actions = [
            #     [ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)
            # ]
            # next_obs, rewards, dones, infos = env.step(actions)
            # for i, agent in enumerate(env.agent_iter()):
            #     action = maddpg.agent[i].step(torch_obs[i])
            #     action = action.data.numpy()
            #     actions.append(action)
            agent_actions = {}
            for a, ac in zip(env.possible_agents, torch_agent_actions):
                agent_actions[a] = ac.data.numpy().flatten()

            print("obs.keys():", obs.keys())
            print("obs.shape:", list(obs.values())[0].shape)
            print("obs:", obs)
            print("torch_obs:", torch_obs)
            print("actions[0].shape:", list(agent_actions.values())[0].shape)
            print("agent_actions:", agent_actions)
            next_obs, rewards, dones, infos = env.step(agent_actions)
            print("dones:", dones)
            print("rewards:", rewards)
            print("next_obs:", next_obs)

            replay_buffer.push(
                [obs[a] for a in env.possible_agents],
                [agent_actions[a] for a in env.possible_agents],
                [rewards[a] for a in env.possible_agents],
                [next_obs[a] for a in env.possible_agents],
                [dones[a] for a in env.possible_agents],
            )
            obs = next_obs
            t += config.n_rollout_threads
            if (
                len(replay_buffer) >= config.batch_size
                and (t % config.steps_per_update) < config.n_rollout_threads
            ):
                if USE_CUDA:
                    maddpg.prep_training(device="gpu")
                else:
                    maddpg.prep_training(device="cpu")
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(maddpg.nagents):
                        sample = replay_buffer.sample(
                            config.batch_size, to_gpu=USE_CUDA
                        )
                        maddpg.update(sample, a_i, logger=logger)
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device="cpu")
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads
        )
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar("agent%i/mean_episode_rewards" % a_i, a_ep_rew, ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / "incremental", exist_ok=True)
            maddpg.save(run_dir / "incremental" / ("model_ep%i.pt" % (ep_i + 1)))
            maddpg.save(run_dir / "model.pt")

    maddpg.save(run_dir / "model.pt")
    env.close()
    logger.export_scalars_to_json(str(log_dir / "summary.json"))
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument(
        "model_name", help="Name of directory to store " + "model/training contents"
    )
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=25000, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument(
        "--batch_size", default=1024, type=int, help="Batch size for model training"
    )
    parser.add_argument("--n_exploration_eps", default=25000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument(
        "--agent_alg", default="MADDPG", type=str, choices=["MADDPG", "DDPG"]
    )
    parser.add_argument(
        "--adversary_alg", default="MADDPG", type=str, choices=["MADDPG", "DDPG"]
    )
    parser.add_argument("--discrete_action", action="store_true")

    config = parser.parse_args()

    run(config)
