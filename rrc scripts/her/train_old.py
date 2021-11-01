import numpy as np
import gym
import os, sys
from rrc_example_package.her.arguments import get_args
from mpi4py import MPI
from rrc_example_package.her.rl_modules.ddpg_agent import ddpg_agent
import random
import torch

# from rrc_example_package import rearrange_dice_env
from rrc_example_package.her.rl_modules.sac import sac_agent

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""
def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
    return params

def launch(args):
    # create the ddpg_agent
    env = gym.make(args.env_name)
    # set random seeds for reproduce
    seed = args.seed + MPI.COMM_WORLD.Get_rank()
    print('Seed: {}'.format(seed))
    env.seed(seed)
    print('Seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    # get the environment parameters
    env_params = get_env_params(env)
    # # create the ddpg agent to interact with the environment 
    # ddpg_trainer = ddpg_agent(args, env, env_params)
    # ddpg_trainer.learn()
    if args.algo == 'ddpg':
        # create the ddpg agent to interact with the environment 
        ddpg_trainer = ddpg_agent(args, env, env_params, seed=seed)
        ddpg_trainer.learn()
    elif args.algo == 'sac':
        sac_trainer = sac_agent(args, env, env_params, seed=seed,
                                ac_kwargs=dict(hidden_sizes=[args.hidden_size]*args.n_hiddens))
        sac_trainer.learn()
    else:
        raise NotImplementedError()

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    launch(args)
