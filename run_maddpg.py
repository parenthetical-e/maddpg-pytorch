import argparse

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
from stable_baselines3.common.utils import get_device

from pettingzoo import mpe
from supersuit import stable_baselines3_vec_env_v0
from supersuit import gym_vec_env_v0
from supersuit import pettingzoo_env_to_vec_env_v1
from supersuit import clip_actions_v0

# USE_CUDA = False  # torch.cuda.is_available()


def run(config):
    # --- Seeds etc
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = get_device(config.device)
    if device == "cpu":
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

    print(f"----- Running: {config.env_id} ------")
    print(f"device: {device}")
    print(f"log_dir: {log_dir}")
    
    # --- Build the env, an MPE zoo
    # TODO: add env_hparams as an arg

    Env = getattr(mpe, config.env_id)
    env = Env.parallel_env(continuous_actions=True)
    env = clip_actions_v0(env)
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
    replay_buffer = ReplayBuffer(
        config.buffer_length,
        maddpg.nagents,
        [obsp.shape[0] for obsp in observation_space],
        [acsp.shape[0] if isinstance(acsp, Box) else acsp.n for acsp in action_space],
    )

    # debug
    # for a, ma in zip(env.possible_agents, maddpg.agents):
    #     print(f"{a}.policy", ma.policy)
    #     print(f"{a}.critic", ma.critic)
    #     print(f"{a}.target_policy", ma.target_policy)
    #     print(f"{a}.target_critic", ma.target_critic)

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

        explr_pct_remaining = (
            max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        )
        maddpg.scale_noise(
            config.final_noise_scale
            + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining
        )
        maddpg.reset_noise()

        # --- Rollout loop - we do for a fixed length
        for et_i in range(config.episode_length):

            # If there are no agents the env
            # should be restarted.
            if len(env.agents) == 0:
                obs = env.reset()

            # rearrange observations for maddpg
            torch_obs = [
                torch.tensor(obs[a], requires_grad=False).unsqueeze(0)
                for a in env.possible_agents
            ]
            # Agents act 'at once' (as parallelized AEC)
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # rearrange actions for zoo environment
            agent_actions = {}
            for a, ac in zip(env.possible_agents, torch_agent_actions):
                # The env will only accept 'env.possible_agents',
                # so we filter out the dead
                if a in env.possible_agents:
                    agent_actions[a] = ac.data.numpy().flatten()
            # !
            next_obs, rewards, dones, infos = env.step(agent_actions)
            replay_buffer.push(
                [obs[a] for a in env.possible_agents],
                [agent_actions[a] for a in env.possible_agents],
                [rewards[a] for a in env.possible_agents],
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
                maddpg.prep_training(device=device)
                # if USE_CUDA:
                #     maddpg.prep_training(device="gpu")
                # else:
                #     maddpg.prep_training(device="cpu")
                for u_i in range(config.n_rollout_threads):
                    # When `maddpg` is built it uses the env and
                    # env.possible_agents to setup its own agents
                    # BUT the agents are indexed by ints not keys
                    # so we need to loop over all possible.
                    for a_i, a_n in enumerate(env.possible_agents):
                        sample = replay_buffer.sample(config.batch_size, device)
                        maddpg.update(sample, a_i, a_n, logger=logger)
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device="cpu")

        # --- Post-episode LOG....
        # Data
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

        # Models
        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / "incremental", exist_ok=True)
            maddpg.save(run_dir / "incremental" / ("model_ep%i.pt" % (ep_i + 1)))
            maddpg.save(run_dir / "model.pt")

    # --- Cleanup/save/etc are done
    maddpg.save(run_dir / "model.pt")
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
    parser.add_argument(
        "--agent_alg", default="MADDPG", type=str, choices=["MADDPG", "DDPG"]
    )
    parser.add_argument(
        "--adversary_alg", default="MADDPG", type=str, choices=["MADDPG", "DDPG"]
    )
    parser.add_argument("--discrete_action", action="store_true")
    parser.add_argument("--device", help="Set device", default="cpu", type=str)

    config = parser.parse_args()

    run(config)
