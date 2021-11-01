"""
TODO:
    - Implement tunable alpha
    - Address any issues: https://github.com/openai/spinningup/issues?q=is%3Aissue+is%3Aopen+sac
    - ensure actions are squashed to correct range
    
"""


from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import rrc_example_package.her.rl_modules.sac_core as core
# from spinup.utils.logx import EpochLogger

from rrc_example_package.her.her_modules.her import her_sampler
from rrc_example_package.her.rl_modules.replay_buffer import replay_buffer
from rrc_example_package.her.mpi_utils.normalizer import normalizer
from mpi4py import MPI
from rrc_example_package.her.mpi_utils.mpi_utils import sync_networks, sync_grads
import os
from datetime import datetime


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}



class sac_agent:
    def __init__(self, args, env, env_params, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), 
                 seed=0, logger_kwargs=dict()
                 ):
        """
        Soft Actor-Critic (SAC)
        Args:
            env_fn : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.
            actor_critic: The constructor method for a PyTorch Module with an ``act`` 
                method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
                The ``act`` method and ``pi`` module should accept batches of 
                observations as inputs, and ``q1`` and ``q2`` should accept a batch 
                of observations and a batch of actions as inputs. When called, 
                ``act``, ``q1``, and ``q2`` should return:
                ===========  ================  ======================================
                Call         Output Shape      Description
                ===========  ================  ======================================
                ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                               | observation.
                ``q1``       (batch,)          | Tensor containing one current estimate
                                               | of Q* for the provided observations
                                               | and actions. (Critical: make sure to
                                               | flatten this!)
                ``q2``       (batch,)          | Tensor containing the other current 
                                               | estimate of Q* for the provided observations
                                               | and actions. (Critical: make sure to
                                               | flatten this!)
                ===========  ================  ======================================
                Calling ``pi`` should return:
                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                               | given observations.
                ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                               | actions in ``a``. Importantly: gradients
                                               | should be able to flow back into ``a``.
                ===========  ================  ======================================
            ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
                you provided to SAC.
            seed (int): Seed for random number generators.
            steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
                for the agent and the environment in each epoch.
            epochs (int): Number of epochs to run and train agent.
            replay_size (int): Maximum length of replay buffer.
            gamma (float): Discount factor. (Always between 0 and 1.)
            polyak (float): Interpolation factor in polyak averaging for target 
                networks. Target networks are updated towards main networks 
                according to:
                .. math:: \\theta_{\\text{targ}} \\leftarrow 
                    \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta
                where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
                close to 1.)
            lr (float): Learning rate (used for both policy and value learning).
            alpha (float): Entropy regularization coefficient. (Equivalent to 
                inverse of reward scale in the original SAC paper.)
            batch_size (int): Minibatch size for SGD.
            start_steps (int): Number of steps for uniform-random action selection,
                before running real policy. Helps exploration.
            update_after (int): Number of env interactions to collect before
                starting to do gradient descent updates. Ensures replay buffer
                is full enough for useful updates.
            update_every (int): Number of env interactions that should elapse
                between gradient descent updates. Note: Regardless of how long 
                you wait between updates, the ratio of env steps to gradient steps 
                is locked to 1.
            num_test_episodes (int): Number of episodes to test the deterministic
                policy at the end of each epoch.
            max_ep_len (int): Maximum length of trajectory / episode / rollout.
            logger_kwargs (dict): Keyword args for EpochLogger.
            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.
        """
        
        self.args = args
        # self.logger = EpochLogger(**logger_kwargs)
        # self.logger.save_config(locals())
    
        torch.manual_seed(seed)
        np.random.seed(seed)
    
        self.env, self.test_env = env, env
        self.env_params = env_params
        
        # # Action limit for clamping: critically, assumes all dimensions share the same bound!
        # act_limit = env.action_space.high[0]
    
        # Create actor-critic module and target networks
        self.ac = actor_critic(self.env_params, **ac_kwargs)
        # sync the networks across the cpus
        sync_networks(self.ac)
        self.ac_targ = deepcopy(self.ac)
    
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
            
        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters()) # TODO: fix!!!!
        # self.q_params = [x for x in self.ac.q1.parameters()] + [x for x in self.ac.q2.parameters()]
    
        # # Experience buffer
        # replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward, self.args.seperate_her)
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        # create the normalizer
        self.o_norm = normalizer(size=self.env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=self.env_params['goal'], default_clip_range=self.args.clip_range)
    
        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('pi, q1, q2 variables: {}\n'.format(var_counts))
        # self.logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)
        
        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.args.lr_actor)
        self.q_optimizer = Adam(self.q_params, lr=self.args.lr_critic)
    
        # create the dict for store the model
        if MPI.COMM_WORLD.Get_rank() == 0:
            self.args.save_dir += self.args.algo
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            # path to save the model
            self.model_path = os.path.join(self.args.save_dir, self.args.exp_dir)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
       
        # Setup logs
        self.logs = {
                    'Epoch': [],
                    'ExploitSuccess': [],
                    'ExploreSuccess': [],
                    'Q1': [],
                    'Q2': [],
                    'LogPi': [],
                    'LossPi': [],
                    'LossQ': [],
                    }
    
        # # Set up model saving
        # self.logger.setup_pytorch_saver(self.ac)

    def learn(self):
        """
        train the network

        """
        for epoch in range(self.args.n_epochs):
            explore_success = []
            stats = {'Q1': [], 'Q2': [], 'LogPi': [], 'LossPi': [], 'LossQ': []}
            for _ in range(self.args.n_cycles):
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                for _ in range(self.args.num_rollouts_per_mpi):
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                    # reset the environment
                    observation = self.env.reset()
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            if epoch > self.args.start_epochs:
                                input_tensor = self.preproc_inputs(obs, g)
                                action = self.get_action(input_tensor)
                            else:
                                action = self.env.action_space.sample()
                        # feed the actions into the environment
                        observation_new, r, _, info = self.env.step(action)
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']
                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action.copy())
                        # re-assign the observation
                        obs = obs_new
                        ag = ag_new
                    explore_success.append(r)
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)
                    # # End of trajectory handling
                    # self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
                # store the episodes
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                self.update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
                for _ in range(self.args.n_batches):
                    update_stats = self.update()
                    for i, key in enumerate(stats.keys()):
                        stats[key].append(update_stats[i])
            # Calc explore success rate
            explore_success = np.mean(explore_success)
            explore_success = MPI.COMM_WORLD.allreduce(explore_success, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
            explore_success += 1
            # Test the performance of the deterministic version of the agent.
            exploit_success = self.test_agent()
            # Update logs
            for i, key in enumerate(stats.keys()):
                stats[key] = np.mean(stats[key])
            # TODO: clean up this mess
            self.update_logs([epoch, exploit_success, explore_success, stats['Q1'], stats['Q2'], stats['LogPi'], stats['LossPi'], stats['LossQ']])
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('[{}] epoch: {} exploit: {:.2f} explore: {:.2f} q1: {:.2f} q2: {:.2f} logpi: {:.2f} losspi: {:.2f} lossq: {:.2f}'
                      .format(datetime.now(), epoch, exploit_success, explore_success, stats['Q1'], stats['Q2'], stats['LogPi'], stats['LossPi'], stats['LossQ']))
                np.save(self.model_path + '/logs.npy', self.logs)
                if self.args.save_models and (epoch % self.args.save_freq) == 0:
                    torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.ac.state_dict()], \
                                self.model_path + '/ac_epoch{}.pt'.format(epoch))
                        
    def update_logs(self, stats):
        for i, key in enumerate(self.logs.keys()):
            self.logs[key].append(stats[i])       

    def get_action(self, input_tensor, deterministic=False):
        return np.squeeze(self.ac.act(input_tensor, deterministic))
    
    def preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # pre_process the inputs
    def preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs

    # update the normalizer
    def update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[0] * mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs, 
                       'ag': mb_ag,
                       'g': mb_g, 
                       'actions': mb_actions, 
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self.preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def update(self):
        transitions = self.buffer.sample(self.args.batch_size)
        data = self.process_update_data(transitions)
        
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        sync_grads(self.ac.q1)
        sync_grads(self.ac.q2)
        self.q_optimizer.step()

        # # Record things
        # self.logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        sync_grads(self.ac.pi)
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next SAC step.
        for p in self.q_params:
            p.requires_grad = True

        # # Record things
        # self.logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.args.polyak_sac)
                p_targ.data.add_((1 - self.args.polyak_sac) * p.data)
                
        return [q_info['Q1Vals'], q_info['Q2Vals'], pi_info['LogPi'], loss_pi.item(), loss_q.item()]

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data):
        o, a, r, o2 = data['obs'], data['act'], data['rew'], data['obs2'] #, data['done']
        # WARNING: always setting 'done' to False
        d = 0

        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            # TODO: clip target?
            backup = r + self.args.gamma * (1 - d) * (q_pi_targ - self.args.alpha * logp_a2) # why is alpha negative??

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        o = data['obs']
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        
        # Entropy-regularized policy loss
        loss_pi = (self.args.alpha * logp_pi - q_pi).mean()
        # print('loss_pi: {}'.format(loss_pi))
        
        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())
        
        return loss_pi, pi_info

    def process_update_data(self, transitions):
        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self.preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self.preproc_og(o_next, g)
        # WARNING: not including any within episode goal changes
        # Normalize obs, g
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # convert to tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32).squeeze()
        return {'obs': inputs_norm_tensor, 'act': actions_tensor, 'rew': r_tensor, 'obs2': inputs_next_norm_tensor}

   
    # do the evaluation
    def test_agent(self):
        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                # self.env.render()
                with torch.no_grad():
                    input_tensor = self.preproc_inputs(obs, g)
                    actions = self.get_action(input_tensor)
                observation_new, r, _, info = self.env.step(actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(r)
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return 1 + (global_success_rate / MPI.COMM_WORLD.Get_size())

    # # Prepare for interaction with environment
    # total_steps = steps_per_epoch * epochs
    # start_time = time.time()
    # o, ep_ret, ep_len = env.reset(), 0, 0

    # # Main loop: collect experience in env and update/log each epoch
    # for t in range(total_steps):
        
    #     # Until start_steps have elapsed, randomly sample actions
    #     # from a uniform distribution for better exploration. Afterwards, 
    #     # use the learned policy. 
    #     if t > start_steps:
    #         a = get_action(o)
    #     else:
    #         a = env.action_space.sample()

    #     # Step the env
    #     o2, r, d, _ = env.step(a)
    #     ep_ret += r
    #     ep_len += 1

    #     # Ignore the "done" signal if it comes from hitting the time
    #     # horizon (that is, when it's an artificial terminal signal
    #     # that isn't based on the agent's state)
    #     d = False if ep_len==max_ep_len else d

    #     # Store experience to replay buffer
    #     replay_buffer.store(o, a, r, o2, d)

    #     # Super critical, easy to overlook step: make sure to update 
    #     # most recent observation!
    #     o = o2
        
    #     # End of trajectory handling
    #     if d or (ep_len == max_ep_len):
    #         logger.store(EpRet=ep_ret, EpLen=ep_len)
    #         o, ep_ret, ep_len = env.reset(), 0, 0

    #     # Update handling
    #     if t >= update_after and t % update_every == 0:
    #         for j in range(update_every):
    #             batch = replay_buffer.sample_batch(batch_size)
    #             update(data=batch)

    #     # End of epoch handling
    #     if (t+1) % steps_per_epoch == 0:
    #         epoch = (t+1) // steps_per_epoch

    #         # Save model
    #         if (epoch % save_freq == 0) or (epoch == epochs):
    #             logger.save_state({'env': env}, None)

    #         # Test the performance of the deterministic version of the agent.
    #         test_agent()

    #         # Log info about epoch
    #         logger.log_tabular('Epoch', epoch)
    #         logger.log_tabular('EpRet', with_min_and_max=True)
    #         logger.log_tabular('TestEpRet', with_min_and_max=True)
    #         logger.log_tabular('EpLen', average_only=True)
    #         logger.log_tabular('TestEpLen', average_only=True)
    #         logger.log_tabular('TotalEnvInteracts', t)
    #         logger.log_tabular('Q1Vals', with_min_and_max=True)
    #         logger.log_tabular('Q2Vals', with_min_and_max=True)
    #         logger.log_tabular('LogPi', with_min_and_max=True)
    #         logger.log_tabular('LossPi', average_only=True)
    #         logger.log_tabular('LossQ', average_only=True)
    #         logger.log_tabular('Time', time.time()-start_time)
    #         logger.dump_tabular()
            
    
    # def test_agent():
    #     for j in range(num_test_episodes):
    #         o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
    #         while not(d or (ep_len == max_ep_len)):
    #             # Take deterministic actions at test time 
    #             o, r, d, _ = test_env.step(get_action(o, True))
    #             ep_ret += r
    #             ep_len += 1
    #         logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
    

# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--env', type=str, default='HalfCheetah-v2')
#     parser.add_argument('--hid', type=int, default=256)
#     parser.add_argument('--l', type=int, default=2)
#     parser.add_argument('--gamma', type=float, default=0.99)
#     parser.add_argument('--seed', '-s', type=int, default=0)
#     parser.add_argument('--epochs', type=int, default=50)
#     parser.add_argument('--exp_name', type=str, default='sac')
#     args = parser.parse_args()

#     from spinup.utils.run_utils import setup_logger_kwargs
#     logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

#     torch.set_num_threads(torch.get_num_threads())

#     sac(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
#         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
#         gamma=args.gamma, seed=args.seed, epochs=args.epochs,
#         logger_kwargs=logger_kwargs)
