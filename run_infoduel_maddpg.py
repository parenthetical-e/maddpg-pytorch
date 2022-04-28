import argparse
from curses import meta
import imp
from multiprocessing.spawn import import_main_path

# import imp
import torch

# import time
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from infoduel_maddpg.utils.buffer import ReplayBuffer
from infoduel_maddpg.core import MADDPG

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from pettingzoo import mpe
from supersuit import stable_baselines3_vec_env_v0
from supersuit import gym_vec_env_v0
from supersuit import pettingzoo_env_to_vec_env_v1
from supersuit import clip_actions_v0

from dualer.wrappers.academic import StatePrediction
from dualer.wrappers.normalize import FoldChangeReward
from dualer.wrappers.normalize import MovingFoldChangeReward

USE_CUDA = False  # torch.cuda.is_available()


def make_env(env_id, n_rollout_threads, seed, discrete_action=False):
    # Ugly:
    # A closure dance (for some reason) to init the env
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
    # --- Seeds etc
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)

    # --- Setup log paths.
    #
    # Do not delete data, but add INT suffixs ala,
    #
    # ./models/env_id/model_name/run_INT
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

    # --- Build the env, an MPE zoo
    # TODO: add env_hparams as an arg

    Env = getattr(mpe, config.env_id)
    env = Env.parallel_env(continuous_actions=True)
    env = clip_actions_v0(env)
    env = StatePrediction(env)  # generates intrinsic/reward
    env.reset()

    # Make access to 'space' info in format that is
    # expected through the codebase
    action_space = [env.action_space(a) for a in env.possible_agents]
    observation_space = [env.observation_space(a) for a in env.possible_agents]

    # --- Build the model and what it needs to learn (buffers)
    maddpg = MADDPG.init_from_env(
        env,
        agent_alg=config.agent_alg,
        adversary_alg=config.adversary_alg,
        tau=config.tau,
        lr=config.lr,
        hidden_dim=config.hidden_dim,
    )
    intrinsic_maddpg = MADDPG.init_from_env(
        env,
        agent_alg=config.agent_alg,
        adversary_alg=config.adversary_alg,
        tau=config.tau,
        lr=config.lr,
        hidden_dim=config.hidden_dim,
    )
    replay_buffer = ReplayBuffer(
        config.buffer_length,
        maddpg.nagents,
        [obsp.shape[0] for obsp in observation_space],
        [acsp.shape[0] if isinstance(acsp, Box) else acsp.n for acsp in action_space],
    )
    intrinsic_replay_buffer = ReplayBuffer(
        config.buffer_length,
        maddpg.nagents,
        [obsp.shape[0] for obsp in observation_space],
        [acsp.shape[0] if isinstance(acsp, Box) else acsp.n for acsp in action_space],
    )

    # ---
    # RUN loop: run a set of episodes (n_episodes), where each has a
    # pre-definited duration (episode_length)
    t = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print(
            f"Episodes {ep_i + 1}-{ep_i + 1 + config.n_rollout_threads} of {config.n_episodes}"
        )

        # Reset several things:
        # - env.
        # - buffers,
        # - and the exploration noise (if it was used).
        obs = env.reset()
        maddpg.prep_rollouts(device="cpu")

        # explr_pct_remaining = (
        #     max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        # )
        # maddpg.scale_noise(
        #     config.final_noise_scale
        #     + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining
        # )
        # maddpg.reset_noise()

        # --- Do the INFODUEL!
        # Set the policy on a per episode basis for a little more
        # stability then on every step. This is violation of out
        # formality, but only a little one. :)
        meta_maddpg = {}
        for a in env.agents:
            # Get best from last episode. The best is what we duel with!
            last_reward = replay_buffer.rew_buffs[a].max()
            last_intrinsic = intrinsic_replay_buffer.rew_buffs[a].max()

            # Use best to set the policy, agent by agent.
            # They can either persue rewards or info value,
            # but never both at the same time...
            #
            # This is why
            # we call this library 'INFODUEL'
            if last_reward >= (last_intrinsic - config.eta):
                meta_maddpg[a] = maddpg
            else:
                meta_maddpg[a] = intrinsic_maddpg

        # --- Rollout loop - we do for a fixed length
        for et_i in range(config.episode_length):
            # rearrange observations for maddpg
            torch_obs = [
                torch.tensor(obs[a], requires_grad=False).unsqueeze(0)
                for a in env.agents
            ]
            agent_actions = {}
            for i, a in enumerate(env.agents):
                act = meta_maddpg[a].agent.step(torch_obs[i])
                act = act.data.numpy().flatten()
                agent_actions[a] = act

            print(">>>>> Episode start <<<<<<<<")
            print("last_reward:", last_reward)
            print("last_intrinsic:", last_intrinsic)
            print("last_intrinsic (adj):", last_intrinsic - config.eta)
            print("meta_maddpg: ", meta_maddpg)

            # Agents act 'at once' (as parallelized AEC)
            # torch_agent_actions = maddpg.step(torch_obs, explore=True)

            # rearrange actions for zoo environment
            # agent_actions = {}
            # for a, ac in zip(env.possible_agents, torch_agent_actions):
            #     agent_actions[a] = ac.data.numpy().flatten()

            # !
            next_obs, _, dones, infos = env.step(agent_actions)
            # We wraaped the env in StatePrediction, a kind of
            # acamemic wrapper that normally returns some user
            # set mixture of intrinsix rewards. However the
            # 'infos' contains the pure reward and intrinsic, so
            # we extact and buffer them independently.
            rewards = infos["env_reward"]
            intrinsics = infos["intrinsic_reward"]
            replay_buffer.push(
                [obs[a] for a in env.possible_agents],
                [agent_actions[a] for a in env.possible_agents],
                [rewards[a] for a in env.possible_agents],
                [next_obs[a] for a in env.possible_agents],
                [dones[a] for a in env.possible_agents],
            )
            intrinsic_replay_buffer.push(
                [obs[a] for a in env.possible_agents],
                [agent_actions[a] for a in env.possible_agents],
                [intrinsics[a] for a in env.possible_agents],
                [next_obs[a] for a in env.possible_agents],
                [dones[a] for a in env.possible_agents],
            )

            # setup for next step
            obs = next_obs
            t += config.n_rollout_threads

            # print("----------------------------")
            # print("ei_i", et_i)
            # print("obs.keys():", obs.keys())
            # print("obs.shape:", list(obs.values())[0].shape)
            # print("obs:", obs)
            # print("torch_obs:", torch_obs)
            # print("actions[0].shape:", list(agent_actions.values())[0].shape)
            # print("agent_actions:", agent_actions)
            # print("torch_agent_actions:", torch_agent_actions)
            # print("dones:", dones)
            # print("rewards:", rewards)
            # print("next_obs:", next_obs)

            # train (ugly code)
            if (
                len(replay_buffer) >= config.batch_size
                and (t % config.steps_per_update) < config.n_rollout_threads
            ):
                # Reward
                if USE_CUDA:
                    maddpg.prep_training(device="gpu")
                else:
                    maddpg.prep_training(device="cpu")
                for u_i in range(config.n_rollout_threads):
                    # When `maddpg` is built it uses the env and
                    # env.possible_agents to setup itss own agents
                    # BUT the agents are indexed by ints not keys
                    # so wee need to loop over all possible but only
                    # learn from current env.agents. This prevents
                    # 'overtraining' on stale experiende in dead
                    # agents who may never the less later revive.
                    for a_i, a_n in enumerate(env.possible_agents):
                        if a_n in env.agents:
                            sample = replay_buffer.sample(
                                config.batch_size, to_gpu=USE_CUDA
                            )
                            maddpg.update(sample, a_i, a_n, logger=logger)
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device="cpu")
                # Intrinsic
                if USE_CUDA:
                    intrinsic_maddpg.prep_training(device="gpu")
                else:
                    intrinsic_maddpg.prep_training(device="cpu")
                for u_i in range(config.n_rollout_threads):
                    for a_i, a_n in enumerate(env.possible_agents):
                        if a_n in env.agents:
                            sample = replay_buffer.sample(
                                config.batch_size, to_gpu=USE_CUDA
                            )
                            intrinsic_maddpg.update(sample, a_i, a_n, logger=logger)
                    intrinsic_maddpg.update_all_targets()
                intrinsic_maddpg.prep_rollouts(device="cpu")

        # --- Post-episode LOG....
        # Data
        # reward
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads
        )
        for i, a in enumerate(env.possible_agents):
            logger.add_scalar(
                f"{a}/std_episode_actions", replay_buffer.ac_buffs[i].std(), ep_i
            )
            logger.add_scalar(
                f"{a}/std_episode_rewards", replay_buffer.rew_buffs[i].std(), ep_i
            )
            logger.add_scalar(f"{a}/mean_episode_rewards", ep_rews[i], ep_i)
        ep_intrins = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads
        )
        # intrinsic
        for i, a in enumerate(env.possible_agents):
            logger.add_scalar(
                f"{a}/std_episode_actions_intrinsic",
                intrinsic_replay_buffer.ac_buffs[i].std(),
                ep_i,
            )
            logger.add_scalar(
                f"{a}/std_episode_intrinsic",
                intrinsic_replay_buffer.rew_buffs[i].std(),
                ep_i,
            )
            logger.add_scalar(f"{a}/mean_episode_intrinsic", ep_intrins[i], ep_i)

        # Models
        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / "incremental", exist_ok=True)
            # reward
            maddpg.save(run_dir / "incremental" / (f"model_ep{ep_i}.pt"))
            maddpg.save(run_dir / "model.pt")
            # intrinsic
            intrinsic_maddpg.save(
                run_dir / "incremental" / (f"intrinsic_model_ep{ep_i}.pt")
            )
            intrinsic_maddpg.save(run_dir / "intrinsic_model.pt")

    # --- Cleanup/save/etc are done
    maddpg.save(run_dir / "model.pt")
    intrinsic_maddpg.save(run_dir / "intrinsic_model.pt")
    env.close()
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
    parser.add_argument("--eta", default=1.0, type=float)
    parser.add_argument(
        "--agent_alg", default="MADDPG", type=str, choices=["MADDPG", "DDPG"]
    )
    parser.add_argument(
        "--adversary_alg", default="MADDPG", type=str, choices=["MADDPG", "DDPG"]
    )
    parser.add_argument("--discrete_action", action="store_true")

    config = parser.parse_args()

    run(config)
