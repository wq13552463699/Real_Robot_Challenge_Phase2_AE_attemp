import numpy as np
import gym
import os, sys
from rrc_example_package.her.arguments import get_args
from mpi4py import MPI
# from rrc_example_package.her.rl_modules.ddpg_agent import ddpg_agent
from rrc_example_package.her.rl_modules.ddpg_agent_rrc import ddpg_agent_rrc
import random
import torch

from rrc_example_package import cube_trajectory_env

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

def main():
    # get the params
    args = get_args()
    # args.n_cycles = 1
    # args.n_batches = 1
    args.n_epochs = 300
    # args.num_rollouts_per_mpi = 2
    # args.batch_size = 512
    
    # create the ddpg_agent
    env = cube_trajectory_env.CustomSimCubeEnv()
    # set random seeds for reproduce
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    # get the environment parameters
    env_params = get_env_params(env)
    # create the ddpg agent to interact with the environment 
    ddpg_trainer = ddpg_agent_rrc(args, env, env_params)
    ddpg_trainer.learn()

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    main()
