import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from rrc_example_package.her.mpi_utils.mpi_utils import sync_networks, sync_grads
from rrc_example_package.her.rl_modules.replay_buffer import replay_buffer
from rrc_example_package.her.rl_modules.models import actor, critic, DynamicsModel
from rrc_example_package.her.mpi_utils.normalizer import normalizer
from rrc_example_package.her.her_modules.her import her_sampler

import torch.nn as nn

"""
ddpg with HER (MPI-version)

"""
class ddpg_agent_rrc:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
        self.env_params = env_params
        # create the network
        self.actor_network = actor(env_params)
        self.critic_network = critic(env_params)
        
        # load in pretrained networks (trained in difficulty 1)
        model_path = args.save_dir + 'pretrained/model131.pt'
        o_mean, o_std, g_mean, g_std, actor_state, critic_state = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.actor_network.load_state_dict(actor_state)
        self.critic_network.load_state_dict(critic_state)
        
        # Create dynamics model
        self.dynamics_model = DynamicsModel(env_params['obs'], env_params['action'], hiddens=3).float()
        self.dm_optimizer = torch.optim.Adam(self.dynamics_model.parameters(), lr=1e-3)
        self.mseloss = nn.MSELoss().float()
        
        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        # build up the target network
        self.actor_target_network = actor(env_params)
        self.critic_target_network = critic(env_params)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        # if use gpu
        if self.args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward)
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)
        self.delta_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        
        # Load in pretrained norms
        self.o_norm.mean, self.o_norm.std = o_mean, o_std
        self.g_norm.mean, self.g_norm.std = g_mean, g_std
        
        # create the dict for store the model
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            # path to save the model
            self.model_path = os.path.join(self.args.save_dir, 'rrc_run3')
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)

    def learn(self):
        """
        train the network

        """
        print('Beginning HER training')
        # Fill buffers with some experience to prevent catastophic forgetting from initialised controller
        self._collect_exp(rollouts=100, difficulty=1)
        self.update_dynamics(steps=int(10000))
        # start to collect samples
        for epoch in range(self.args.n_epochs):
            actor_loss = []
            dy_loss = []
            r_intrinsics = []
            for _ in range(self.args.n_cycles):
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                # print('[{}] beginning rollouts'.format(datetime.now()))
                for _ in range(self.args.num_rollouts_per_mpi):
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                    # reset the environment
                    observation = self.env.reset(difficulty=self.sample_difficulty())
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            input_tensor = self._preproc_inputs(obs, g)
                            pi = self.actor_network(input_tensor)
                            action = self._select_actions(pi)
                        # feed the actions into the environment
                        observation_new, _, _, info = self.env.step(action)
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
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)
                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
                # store the episodes
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
                # print('[{}] beginning updates'.format(datetime.now()))
                for _ in range(self.args.n_batches):
                    # train the network
                    a_loss, ri = self._update_network()
                    actor_loss += [a_loss]
                    r_intrinsics += [ri]
                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)
                # update dynamics model
                dy_loss += [self.update_dynamics(steps=5)]
            # start to do the evaluation
            # print('[{}] beginning evaluation'.format(datetime.now()))
            success_rate = self._eval_agent()
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('[{}] epoch is: {}, success rate: {:.3f} , a_loss, {:.2f} , dy_loss: {:.3f} , ri: {:.3f}'.format(datetime.now(), epoch, success_rate, np.mean(actor_loss), np.mean(dy_loss), np.mean(r_intrinsics)))
                torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict(), self.critic_network.state_dict()], \
                            self.model_path + '/ac_model{}.pt'.format(epoch))
                torch.save([self.o_norm.mean, self.o_norm.std, self.delta_norm.mean, self.delta_norm.std, self.dynamics_model.state_dict()], \
                           self.model_path + '/dy_model{}.pt'.format(epoch))

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs
    
    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                            size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
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
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # delta calculation
        mb_obs = np.clip(mb_obs, -self.args.clip_obs, self.args.clip_obs)
        mb_delta = mb_obs[:,1:,:] - mb_obs[:,:-1,:]
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        self.delta_norm.update(mb_delta)
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()
        self.delta_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)
        # Add intrinsic reward
        r_intrinsic = self.get_intrinsic_reward(transitions['obs'], transitions['actions'], transitions['obs_next'])
        transitions['r'] += r_intrinsic
        ri = np.mean(r_intrinsic)
        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32) 
        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()
        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma) #TODO: fix this!!!!!!!!!!!!!!
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        # the q loss
        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # the actor loss
        actions_real = self.actor_network(inputs_norm_tensor)
        actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network)
        self.critic_optim.step()
        
        return actor_loss.detach().numpy(), ri
    
    def get_intrinsic_reward(self, obs, a, obs_next, clip_max=0.8, scale=1):
        delta = obs_next - obs
        obs, delta = self._preproc_og(obs, delta)
        obs_norm = torch.tensor(self.o_norm.normalize(obs))
        delta_norm = self.delta_norm.normalize(delta)
        
        delta_pred = self.dynamics_model(obs_norm, torch.tensor(a, dtype=torch.float32))
        error = scale * np.mean(np.square(delta_pred.detach().numpy() - delta_norm), axis=-1)
        ri = np.expand_dims(np.clip(error, 0, clip_max), axis=-1)
        return ri
    
    def _collect_exp(self, rollouts=100, difficulty=1):
        mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
        for _ in range(rollouts):
            # reset the rollouts
            ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
            # reset the environment
            observation = self.env.reset(difficulty=difficulty)
            obs = observation['observation']
            ag = observation['achieved_goal']
            g = observation['desired_goal']
            # start to collect samples
            for t in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor)
                    action = self._select_actions(pi)
                # feed the actions into the environment
                observation_new, _, _, info = self.env.step(action)
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
            ep_obs.append(obs.copy())
            ep_ag.append(ag.copy())
            mb_obs.append(ep_obs)
            mb_ag.append(ep_ag)
            mb_g.append(ep_g)
            mb_actions.append(ep_actions)
        # convert them into arrays
        mb_obs = np.array(mb_obs)
        mb_ag = np.array(mb_ag)
        mb_g = np.array(mb_g)
        mb_actions = np.array(mb_actions)
        # store the episodes
        self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
        self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
        
    def update_dynamics(self, steps=1, batch_size=512):
        total_loss = 0
        for _ in range(int(steps)):
            # sample the episodes
            transitions = self.buffer.sample(batch_size)
            # pre-process the observation and delta obs
            obs, a, obs_next = transitions['obs'], transitions['actions'], transitions['obs_next']
            delta = obs_next - obs
            obs, delta = self._preproc_og(obs, delta)
            obs_norm = torch.tensor(self.o_norm.normalize(obs), dtype=torch.float32)
            delta_norm = torch.tensor(self.delta_norm.normalize(delta), dtype=torch.float32)
            
            # start to do the update
            self.dm_optimizer.zero_grad()
            delta_pred = self.dynamics_model(obs_norm, torch.tensor(a, dtype=torch.float32))
            loss = self.mseloss(delta_pred, delta_norm)
            loss.backward()
            sync_grads(self.dynamics_model)
            self.dm_optimizer.step()
            
            total_loss += loss.item()
        return total_loss/steps
    
    def sample_difficulty(self, d1_prob=0.2, d2_prob=0.1, d3_prob=0.7):
        difficulty = np.random.choice([1,2,3], p=[d1_prob,d2_prob,d3_prob])
        return difficulty

    # do the evaluation
    def _eval_agent(self):
        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = self.env.reset(difficulty=self.sample_difficulty())
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, _, _, info = self.env.step(actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()