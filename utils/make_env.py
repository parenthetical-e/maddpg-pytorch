"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
    env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""
from operator import ge
from pettingzoo import mpe


def make_env(env_id, benchmark=False, env_kwargs=None):
    """
    Creates a pettingzoo object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        env_id   :   name of the MPE [pettingzoo] env
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)
    """
    # from multiagent.environment import MultiAgentEnv
    # import multiagent.scenarios as scenarios
    Env = getattr(mpe, env_id)
    if env_kwargs is not None:
        env = Env().env()
    else:
        env = Env(**env_kwargs).env()

    # if benchmark:
    #     env = MultiAgentEnv(
    #         world,
    #         scenario.reset_world,
    #         scenario.reward,
    #         scenario.observation,
    #         scenario.benchmark_data,
    #         discrete_action=discrete_action,
    #     )
    # else:
    #     env = MultiAgentEnv(
    #         world,
    #         scenario.reset_world,
    #         scenario.reward,
    #         scenario.observation,
    #         discrete_action=discrete_action,
    #     )

    return env
