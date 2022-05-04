import argparse
import os
import torch
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from infoduel_maddpg.utils.buffer import ReplayBuffer
from infoduel_maddpg.core import MADDPG

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# from stable_baselines3.common.utils import get_device

from pettingzoo import mpe
from supersuit import stable_baselines3_vec_env_v0
from supersuit import gym_vec_env_v0
from supersuit import pettingzoo_env_to_vec_env_v1
from supersuit import clip_actions_v0

from infoduel_maddpg.utils.academic_wrappers import StatePredictionWrapper
from infoduel_maddpg.utils.normalize_wrappers import ClipRewardWrapper
from infoduel_maddpg.utils.normalize_wrappers import MovingFoldChangeRewardWrapper


def run(config):
    # --- Seeds etc
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if config.device == "cpu":
        torch.set_num_threads(config.n_training_threads)

    # --- Setup log paths.
    #
    # Do not delete data, but add INT suffixs ala,
    #
    # ./models/env_id/model_name/run_INT
    model_dir = Path("./models") / config.env_id / config.model_name
    # Generate unique names, `run#``
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
    # Our log at last....
    logger = SummaryWriter(str(log_dir))

    print(f"----- Running: {config.env_id} ------")
    print(f"device: {config.device}")
    print(f"log_dir: {log_dir}")

    # Set
    th_device = torch.device(config.device)

    # --- Build the envs, an MPE zoo
    Env = getattr(mpe, config.env_id)
    # Init
    env = Env.parallel_env(continuous_actions=True)
    env = clip_actions_v0(env)
    academic = Env.parallel_env(continuous_actions=True)
    academic = clip_actions_v0(academic)
    academic = StatePredictionWrapper(
        academic,
        network_hidden=[config.hidden_dim],
        lr=config.lr / 10,
        device=config.device,
    )
    # Fold-change
    env = MovingFoldChangeRewardWrapper(
        env,
        intial_reference_reward=-2,
        bias_reward=0,
    )
    academic = MovingFoldChangeRewardWrapper(
        academic,
        intial_reference_reward=0.001,
        bias_reward=0,
    )
    # Clip
    env = ClipRewardWrapper(env, min_reward=-10, max_reward=10)
    academic = ClipRewardWrapper(academic, min_reward=-10, max_reward=10)

    # Seed
    env.seed(config.seed)
    academic.seed(config.seed)

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
        device=config.device,
    )
    intrinsic_maddpg = MADDPG.init_from_env(
        academic,
        agent_alg=config.agent_alg,
        adversary_alg=config.adversary_alg,
        tau=config.tau,
        lr=config.lr,
        hidden_dim=config.hidden_dim,
        device=config.device,
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
    for n in tqdm(range(config.n_episodes), desc=config.env_id):
        # Reset several things
        obs = env.reset()
        _ = academic.reset()

        maddpg.prep_rollouts()
        intrinsic_maddpg.prep_rollouts()

        # --- Use curious inspiration? (aka parkid)
        inspiration = 0.0
        if config.kappa > 0:
            for i, a in enumerate(env.possible_agents):
                inspiration += (
                    intrinsic_replay_buffer.get_max_rewards(config.episode_length)
                    - config.eta
                )

        # --- Do the INFODUEL!
        # Set the policy on a per episode basis for a little more
        # stability then on every step. This is violation of out
        # formality, but only a little one. :)
        if n == 0:
            last_rewards = [0] * len(env.possible_agents)
            last_intrinsics = [config.eta + 0.0001,] * len(
                academic.possible_agents
            )  # favor first
        else:
            last_rewards = replay_buffer.get_max_rewards(config.episode_length)
            last_intrinsics = intrinsic_replay_buffer.get_max_rewards(
                config.episode_length
            )

        meta_maddpg = {}
        for i, a in enumerate(env.possible_agents):
            # Use best value to set the policy, agent by agent.
            # They can either persue rewards or info value,
            # but never both at the same time...
            #
            # This is why
            # we call this library 'INFODUEL'
            meta = None
            last_reward = last_rewards[i]
            last_intrinsic = last_intrinsics[i] + inspiration
            if last_reward >= (last_intrinsic - config.eta):
                meta_maddpg[a] = maddpg.agents[i]
                meta = 0  # default reward greed
            else:
                meta_maddpg[a] = intrinsic_maddpg.agents[i]
                meta = 1

            # Log here so I don't need to keep track of these
            # last_* and policy values
            logger.add_scalar(f"{a}/policy", meta, n)
            logger.add_scalar(f"{a}/last_reward", last_reward, n)
            logger.add_scalar(
                f"{a}_intrinsic/last_intrinsic",
                last_intrinsic,
                n,
            )
            logger.add_scalar(
                f"{a}_intrinsic/last_intrinsic (adj)",
                last_intrinsic - config.eta,
                n,
            )
            logger.add_scalar(f"{a}_intrinsic/inspiration", inspiration, n)

        # --- Rollout loop
        for _ in range(config.episode_length):
            # If there are no agents, the env
            # should be restarted.
            if len(env.agents) == 0:
                obs = env.reset()
                _ = academic.reset()

            # rearrange observations for maddpg
            torch_obs = [
                torch.tensor(obs[a], requires_grad=False).to(th_device).unsqueeze(0)
                for a in env.agents
            ]

            # Use meta_maddpg to step each agent
            # to get their actions
            agent_actions = {}
            for i, a in enumerate(env.possible_agents):
                act = meta_maddpg[a].step(torch_obs[i], explore=False)
                act = act.data.cpu().numpy().flatten()
                agent_actions[a] = act

            # ...and apply these actions to the env
            next_obs, rewards, dones, infos = env.step(agent_actions)
            next_obs_in, intrinsics, dones_in, infos_in = academic.step(agent_actions)

            replay_buffer.push(
                [obs[a] for a in env.possible_agents],
                [agent_actions[a] for a in env.possible_agents],
                [rewards[a] for a in env.possible_agents],
                [next_obs[a] for a in env.possible_agents],
                [dones[a] for a in env.possible_agents],
            )
            intrinsic_replay_buffer.push(
                [obs[a] for a in academic.possible_agents],
                [agent_actions[a] for a in academic.possible_agents],
                [intrinsics[a] for a in academic.possible_agents],
                [next_obs_in[a] for a in academic.possible_agents],
                [dones_in[a] for a in academic.possible_agents],
            )

            # sanity
            for a in next_obs.keys():
                assert np.allclose(
                    next_obs[a], next_obs_in[a]
                ), f"At step {n}, agent {a} next_obs disagreed between env and academic."

            # setup for next step
            obs = next_obs
            t += 1

            # --- Train (ugly code)?
            if (
                len(replay_buffer) >= config.batch_size
                and (t % config.steps_per_update) == 1
            ):
                # reward
                maddpg.prep_training()
                for i, a in enumerate(env.possible_agents):
                    sample = replay_buffer.sample(config.batch_size, config.device)
                    maddpg.update(sample, i, a, logger=logger)
                maddpg.update_all_targets()
                maddpg.prep_rollouts()
                # info val
                intrinsic_maddpg.prep_training()
                for i, a in enumerate(academic.possible_agents):
                    sample = intrinsic_replay_buffer.sample(
                        config.batch_size, config.device
                    )
                    intrinsic_maddpg.update(sample, i, a, logger=logger)
                intrinsic_maddpg.update_all_targets()
                intrinsic_maddpg.prep_rollouts()

        # --- Post-episode LOG....
        # reward
        for i, a in enumerate(env.possible_agents):
            logger.add_scalar(
                f"{a}/std_episode_actions", replay_buffer.ac_buffs[i].std(), n
            )
            logger.add_scalar(
                f"{a}/std_episode_rewards", replay_buffer.rew_buffs[i].std(), n
            )
            logger.add_scalar(f"{a}/mean_episode_rewards", last_rewards[i], n)
        # intrinsic
        for i, a in enumerate(env.possible_agents):
            logger.add_scalar(
                f"{a}_intrinsic/std_episode_actions",
                intrinsic_replay_buffer.ac_buffs[i].std(),
                n,
            )
            logger.add_scalar(
                f"{a}_intrinsic/std_episode_intrinsic",
                intrinsic_replay_buffer.rew_buffs[i].std(),
                n,
            )
            logger.add_scalar(
                f"{a}_intrinsic/mean_episode_intrinsic", last_intrinsics[i], n
            )

        # Models
        if n % config.save_interval == 1:
            os.makedirs(run_dir / "incremental", exist_ok=True)
            # reward
            maddpg.save(run_dir / "incremental" / (f"model_ep{n}.pt"))
            maddpg.save(run_dir / "model.pt")
            # intrinsic
            intrinsic_maddpg.save(
                run_dir / "incremental" / (f"intrinsic_model_ep{n}.pt")
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
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=25000, type=int)
    parser.add_argument("--n_episodes", default=25000, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument(
        "--batch_size", default=1024, type=int, help="Batch size for model training"
    )
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--eta", default=1.0, type=float, help="Boredom parameter")
    parser.add_argument(
        "--kappa",
        default=0.0,
        type=float,
        help="Info value sharing wieght (>0 activates 'parkid' mode)",
    )
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
