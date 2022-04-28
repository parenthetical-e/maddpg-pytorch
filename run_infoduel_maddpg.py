import argparse
import os
import torch
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

from infoduel_maddpg.utils.academic_wrappers import StatePrediction

USE_CUDA = False  # torch.cuda.is_available()


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
    # we can do without this clip, technically,
    # but a ot of warnings follow, so...
    env = clip_actions_v0(env)

    # generates intrinsic/reward
    # We:
    # - share the size of the hidden layer
    # between academics and actors
    # - Scale academic learning monotonically
    # with actors (the maddpg)
    env = StatePrediction(
        env,
        network_hidden=[config.hidden_dim],
        lr=config.lr / 10.0,
    )
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
        config.info_buffer_length,
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
        intrinsic_maddpg.prep_rollouts(device="cpu")

        # TODO - turn noise back on and port to intrinsix_maddpg?

        # explr_pct_remaining = (
        #     max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        # )
        # maddpg.scale_noise(
        #     config.final_noise_scale
        #     + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining
        # )
        # maddpg.reset_noise()

        # --- Look up the postition in the buffer
        # from last episode (used in the infoduel)
        last_n = replay_buffer.filled_i - config.episode_length
        # print("last_n", last_n)

        # --- Use curious inspiration? (aka parkid)
        inspiration = 0.0
        if config.kappa > 0:
            for i, a in enumerate(env.possible_agents):
                inspiration += intrinsic_replay_buffer.rew_buffs[i][last_n:].max()
            print("inspiration", inspiration)

        # --- Do the INFODUEL!
        # Set the policy on a per episode basis for a little more
        # stability then on every step. This is violation of out
        # formality, but only a little one. :)
        meta_maddpg = {}
        for i, a in enumerate(env.possible_agents):
            if last_n < 1:
                last_reward = 0.0
                last_intrinsic = config.eta + 0.0001  # favor at the start
            else:
                # Get best from last episode.
                # The best is what we duel with!
                last_reward = replay_buffer.rew_buffs[i][last_n:].max()
                last_intrinsic = intrinsic_replay_buffer.rew_buffs[i][last_n:].max()

            # Add contagious curiosity?
            last_intrinsic += inspiration * config.kappa

            # Use best value to set the policy, agent by agent.
            # They can either persue rewards or info value,
            # but never both at the same time...
            #
            # This is why
            # we call this library 'INFODUEL'
            meta = None
            if last_reward >= (last_intrinsic - config.eta):
                meta_maddpg[a] = maddpg.agents[i]
                meta = 0  # default reward greed
            else:
                meta_maddpg[a] = intrinsic_maddpg.agents[i]
                meta = 1

            # Log here so I don't need to keep track of these
            # last_* values
            logger.add_scalar(f"{a}/policy", meta, ep_i)
            logger.add_scalar(f"{a}/last_reward", last_reward, ep_i)
            logger.add_scalar(f"{a}_intrinsic/last_intrinsic", last_intrinsic, ep_i)
            logger.add_scalar(
                f"{a}_intrinsic/last_intrinsic (adj)",
                last_intrinsic - config.eta,
                ep_i,
            )
            logger.add_scalar(f"{a}_intrinsic/inspiration", inspiration, ep_i)

        # --- Rollout loop
        # (we go for a fixed length)
        for et_i in range(config.episode_length):
            # If there are no agents the env
            # should be restarted.
            if len(env.agents) == 0:
                obs = env.reset()

            # rearrange observations for maddpg
            torch_obs = [
                torch.tensor(obs[a], requires_grad=False).unsqueeze(0)
                for a in env.agents
            ]

            # Use meta_maddpg to step each agent
            # to get their actions
            agent_actions = {}
            for i, a in enumerate(env.possible_agents):
                act = meta_maddpg[a].step(torch_obs[i], explore=False)
                act = act.data.numpy().flatten()
                agent_actions[a] = act

            # ...and apply these actions to the env
            next_obs, _, dones, infos = env.step(agent_actions)

            # We wraaped the env in StatePrediction, a kind of
            # acamemic wrapper that normally returns some user
            # set mixture of intrinsic and rewards. However the
            # 'infos' contains the pure reward and intrinsic so
            # we extact and buffer them independently.
            replay_buffer.push(
                [obs[a] for a in env.possible_agents],
                [agent_actions[a] for a in env.possible_agents],
                [infos[a]["env_reward"] for a in env.possible_agents],
                [next_obs[a] for a in env.possible_agents],
                [dones[a] for a in env.possible_agents],
            )
            intrinsic_replay_buffer.push(
                [obs[a] for a in env.possible_agents],
                [agent_actions[a] for a in env.possible_agents],
                [infos[a]["intrinsic_reward"] for a in env.possible_agents],
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
            # print("last_reward:", last_reward)
            # print("last_intrinsic:", last_intrinsic)
            # print("last_intrinsic (adj):", last_intrinsic - config.eta)
            # print("meta_maddpg: ", meta_maddpg)
            # print("agent_actions: ", agent_actions)
            # print("env.possible_agents: ", env.possible_agents)
            # print("env.agents: ", env.agents)
            # print("rewards:", rewards)
            # print("next_obs:", next_obs)
            # print("infos", infos)
            # print("dones:", dones)

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
                # When `maddpg` is built it uses the env and
                # env.possible_agents to setup itss own agents
                # BUT the agents are indexed by ints not keys
                # so wee need to loop over all possible but only
                # learn from current env.agents. This prevents
                # 'overtraining' on stale experiende in dead
                # agents who may never the less later revive.
                for u_i in range(config.n_rollout_threads):
                    for a_i, a_n in enumerate(env.possible_agents):
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
                        sample = intrinsic_replay_buffer.sample(
                            config.batch_size, to_gpu=USE_CUDA
                        )
                        intrinsic_maddpg.update(
                            sample, a_i, a_n + "_intrinsic", logger=logger
                        )
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
        # intrinsic
        ep_intrins = intrinsic_replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads
        )
        for i, a in enumerate(env.possible_agents):
            logger.add_scalar(
                f"{a}_intrinsic/std_episode_actions",
                intrinsic_replay_buffer.ac_buffs[i].std(),
                ep_i,
            )
            logger.add_scalar(
                f"{a}_intrinsic/std_episode_intrinsic",
                intrinsic_replay_buffer.rew_buffs[i].std(),
                ep_i,
            )
            logger.add_scalar(
                f"{a}_intrinsic/mean_episode_intrinsic", ep_intrins[i], ep_i
            )

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
    parser.add_argument("--info_buffer_length", default=int(500), type=int)
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
    parser.add_argument("--eta", default=0.001, type=float, help="Boredom parameter")
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

    config = parser.parse_args()

    run(config)
