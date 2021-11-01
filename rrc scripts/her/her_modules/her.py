import numpy as np

class her_sampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None, seperate_her=False, goal_len=3):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func
        self.seperate_her = seperate_her # TODO: rename as multi-criteria HER
        self.goal_len = goal_len # TODO: make global variable (or other)

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        if self.seperate_her:
            # apply HER individually to each dice
            step = self.goal_len
        else:
            # apply HER across whole goal
            step = transitions['ag'].shape[1]
        # TODO: vectorize
        for i in range(0, transitions['ag'].shape[1], step):
            # her idx
            her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
            future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
            future_offset = future_offset.astype(int)
            future_t = (t_samples + 1 + future_offset)[her_indexes]
            # replace goal with achieved goal
            future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
            # Only replace for selected dice
            transition_goals = transitions['g'][her_indexes]
            transition_goals[:, i:i+step] = future_ag[:, i:i+step]
            transitions['g'][her_indexes] = transition_goals
        # shuffle to ensure Q-function doesn't care about goal order
        transitions['g'] = self.shuffle_goal(transitions['g'])
        # to get the params to re-compute reward
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}
        
        return transitions
    
    def shuffle_goal(self, goals):
        num_dice = int(goals.shape[1]/self.goal_len)
        # TODO: vectorize + simplify
        for i in range(goals.shape[0]):
            goal_idxs = np.arange(num_dice)
            np.random.shuffle(goal_idxs)
            goal_idxs = np.repeat(goal_idxs, self.goal_len)
            goal_idxs *= 3
            add = np.arange(self.goal_len)
            add = np.tile(add, num_dice)
            goal_idxs += add
            goals[i] = goals[i, goal_idxs]
        return goals