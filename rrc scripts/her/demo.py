import torch
from rrc_example_package.her.rl_modules.models import DynamicsModel, actor as ddpg_actor, critic as ddpg_critic
from rrc_example_package.her.arguments import get_args
import gym
import numpy as np

from rrc_example_package import rearrange_dice_env
from rrc_example_package.her.rl_modules.sac_core import MLPActorCritic as sac_ac
from rrc_example_package.her.mpi_utils.normalizer import normalizer

# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs

# this function will choose action for the agent and do the exploration
def select_actions(pi, args, env_params):
    action = pi.cpu().numpy().squeeze()
    # add the gaussian
    action += args.noise_eps * env_params['action_max'] * np.random.randn(*action.shape)
    action = np.clip(action, -env_params['action_max'], env_params['action_max'])
    # random actions...
    random_actions = np.random.uniform(low=-env_params['action_max'], high=env_params['action_max'], \
                                        size=env_params['action'])
    # choose if use the random actions
    action += np.random.binomial(1, args.random_eps, 1)[0] * (random_actions - action)
    return action

def get_sac_action(ac, input_tensor, deterministic=False):
        return np.squeeze(ac.act(input_tensor, deterministic))

def load_ensemble(model_state, env_params, args):
    ensemble_models = []
    for i in range(3):
        dy_model = DynamicsModel(env_params['obs'], env_params['action'])
        # print('Not loading in ensemble!!!')
        dy_model.load_state_dict(model_state['state_dict{}'.format(i)])
        ensemble_models += [dy_model]
    # setup normalisers
    o_norm = normalizer(size=env_params['obs'], default_clip_range=args.clip_range)
    o_norm.mean = model_state['obs_mean']
    o_norm.std = model_state['obs_std']
    delta_norm = normalizer(size=env_params['obs'], default_clip_range=args.clip_range)
    delta_norm.mean = model_state['delta_mean']
    delta_norm.std = model_state['delta_std']
    return ensemble_models, o_norm, delta_norm
    
def get_intrinsic_reward(ensemble, obs, action, o_norm):
    obs = o_norm.normalize(obs)
    predictions = []
    for model in ensemble:
        predictions += [model(torch.tensor(obs), torch.tensor(action)).detach().numpy()]
    predictions = np.clip(np.array(predictions), -5, 5)
    mean_prediction = np.mean(predictions, axis=0)
    dist_from_mean = np.mean(np.square(predictions - mean_prediction), axis=-1)
    r_intrinsic = np.mean(dist_from_mean)
    return np.clip(10 * r_intrinsic, 0, 100)

def get_pred_error(ensemble, obs, action, obs_next, o_norm, delta_normalizer):
    delta = obs_next - obs
    obs_norm = torch.tensor(o_norm.normalize(obs))
    delta_norm = delta_normalizer.normalize(delta)
    delta_pred = ensemble[1](obs_norm, torch.tensor(action, dtype=torch.float32))
    error = np.mean(np.square(delta_pred.detach().numpy() - delta_norm), axis=-1)
    return error

def choose_dice_and_goal(obs, goal_array):
    goal_len = 3
    dice_positions, goals = [], []
    for i in range(27, obs.shape[-1], 3):
        dice_positions.append(obs[i:i+3])
    for i in range(0, goal_array.shape[-1], 3):
        goals.append(goal_array[i:i+3])
    distances = np.zeros((len(dice_positions), len(goals)))
    for i, dice in enumerate(dice_positions):
        for j, goal in enumerate(goals):
            distances[i,j] = np.linalg.norm(dice - goal)
    min_indices = np.where(distances == np.min(distances))
    chosen_dice = dice_positions[min_indices[0][0]]
    chosen_goal = goals[min_indices[1][0]]
    new_obs = obs[0:30]
    new_obs[27:30] = chosen_dice
    new_goal = chosen_goal
    return new_obs, new_goal
    
def get_env_params(env):
    # get the env param
    observation = env.reset()
    # get the environment params
    env_params = {'obs': observation['observation'].shape[0], 
                  'goal': observation['desired_goal'].shape[0], 
                  'action': env.action_space.shape[0], 
                  'action_max': env.action_space.high[0],
                  }
    return observation, env_params
    

def main():
    
    sim=True
    visualization=True
    enable_cameras=False
    num_dice=22
    scale_dice=1
    step_size=50
    image_size=270
    distance_threshold=0.02
    max_steps=30
    demo_length=1
    model_path='src/rrc_example_package/rrc_example_package/her/saved_models/ddpg/increment_dice_add10/acmodel80.pt'
    dy_model_state = torch.load('src/rrc_example_package/rrc_example_package/her/saved_models/ddpg/dice1ri10/Eof3_epoch20.tar')
    include_dice_velocity=False
    include_dice_orient=False
    algo='ddpg'
    choose=False
    explore=False
    include_ri=False
    increment_dice=False        
    extra_dice = 22
    single_dice_focus=True
    
    pre_vis = visualization
    if increment_dice:
        pre_vis = False
    
    args = get_args()
    # create the environment
    env = rearrange_dice_env.SimtoRealRearrangeDiceEnv(
        step_size=step_size,
        sim=sim,
        visualization=pre_vis,
        enable_cameras=enable_cameras,
        num_dice=num_dice,
        image_size=image_size,
        distance_threshold=distance_threshold,
        max_steps=max_steps,
        include_dice_velocity=include_dice_velocity,
        include_dice_orient=include_dice_orient,
        scale_dice=scale_dice,
        single_dice_focus=single_dice_focus
    )
    
    observation, env_params = get_env_params(env)
    print(env_params)
    
    if choose:
        env_params['obs'], env_params['goal'] = 30, 3
    if include_ri:
        ensemble_models, o_norm, delta_norm = load_ensemble(dy_model_state, env_params, args)
    # load the model param
    if algo == 'ddpg':
        o_mean, o_std, g_mean, g_std, model, critic = torch.load(model_path, map_location=lambda storage, loc: storage)
        # create the actor network
        actor_network = ddpg_actor(env_params)
        actor_network.load_state_dict(model)
        actor_network.eval()
        # create the actor network
        critic_network = ddpg_critic(env_params)
        critic_network.load_state_dict(critic)
        critic_network.eval()
        if increment_dice: 
            env = rearrange_dice_env.SimtoRealRearrangeDiceEnv(
                step_size=step_size,
                sim=sim,
                visualization=visualization,
                enable_cameras=enable_cameras,
                num_dice=num_dice,
                image_size=image_size,
                distance_threshold=distance_threshold,
                max_steps=max_steps,
                include_dice_velocity=include_dice_velocity,
                include_dice_orient=include_dice_orient,
                scale_dice=scale_dice,
                single_dice_focus=single_dice_focus
            )
            env.set_num_dice(env.num_dice + extra_dice)
            observation, env_params = get_env_params(env)
            print(env_params)
            print('obs new:')
            print(observation['observation'])
            print('g new:')
            print(observation['desired_goal'])
            actor_network.add_dice(env_params, extra_dice)
            critic_network.add_dice(env_params, extra_dice)
            for _ in range(extra_dice):
                o_mean = np.concatenate((o_mean, o_mean[-3:]))
                o_std = np.concatenate((o_std, o_std[-3:]))
    elif algo == 'sac':
        o_mean, o_std, g_mean, g_std, model = torch.load(model_path, map_location=lambda storage, loc: storage)
        ac = sac_ac(env_params, hidden_sizes=[256]*2)
        ac.load_state_dict(model)
    
    input()
    pred_errors, ris, successes = [], [], []
    for i in range(demo_length):
        if i > 0:
            observation = env.reset()
        # start to do the demo
        obs = observation['observation']
        g = observation['desired_goal']
        for t in range(env._max_episode_steps):
            # env.render()
            if choose:
                obs, g = choose_dice_and_goal(obs, g)
            inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
            with torch.no_grad():
                if algo =='ddpg':
                    pi = actor_network(inputs)
                    q = critic_network(inputs, pi)
                    print('Q: {}'.format(q))
                    if explore:
                        action = select_actions(pi, args, env_params)
                    else:
                        action = pi.detach().numpy().squeeze()
                else:
                    action = get_sac_action(ac, inputs, deterministic= not explore)
            # put actions into the environment
            observation_new, reward, _, info = env.step(action)
            # print(reward)
            if include_ri:
                ri = get_intrinsic_reward(ensemble_models, obs, action, o_norm)
                error = get_pred_error(ensemble_models, obs, action, observation_new['observation'], o_norm, delta_norm)
                pred_errors.append(error), ris.append(ri)
                # print(ri)
            obs = observation_new['observation']
        print('the episode is: {}, is success: {}'.format(i, reward >= 0))
        successes.append(reward >= 0)
    print('Success rate: {}'.format(np.mean(successes)))
    if include_ri:
        print('Mean prediction error: {}'.format(np.mean(pred_errors)))
        print('Mean r_intrinsic: {}'.format(np.mean(ris)))

if __name__ == '__main__':
    main()