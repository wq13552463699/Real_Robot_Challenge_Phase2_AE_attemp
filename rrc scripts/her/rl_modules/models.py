import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.

"""

# define the actor network
class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions
    
    def add_dice(self, env_params, extra_dice):
        with torch.no_grad():
            extra_inputs = extra_dice * 3
            end_shift = env_params['goal']
            temp = self.fc1
            og_ins = np.arange(temp.weight.shape[-1])
            og_ins[-end_shift:] += extra_inputs
            self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
            self.fc1.bias = temp.bias
            self.fc1.weight[:, og_ins] = temp.weight
            self.fc1.weight[:, -(end_shift+extra_inputs):-end_shift] = 0
            

class critic(nn.Module):
    def __init__(self, env_params):
        super(critic, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value
    
    def add_dice(self, env_params, extra_dice):
        with torch.no_grad():
            extra_inputs = extra_dice * 3
            end_shift = env_params['goal'] + env_params['action']
            temp = self.fc1
            ins = np.arange(temp.weight.shape[-1])
            ins[-end_shift:] += extra_inputs
            self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
            self.fc1.bias = temp.bias
            self.fc1.weight[:, ins] = temp.weight
            self.fc1.weight[:, -(end_shift+extra_inputs):-end_shift] = 0
    

# define simple feed-forward dynamics model
class DynamicsModel(nn.Module):
    def __init__(self, obs_size, act_size, hiddens=2, hidden_size=512):
        super(DynamicsModel, self).__init__()
        assert hiddens > 0, "Must have at least 1 hidden layer"
        self.hidden_layers = nn.ModuleList([nn.Linear(obs_size + act_size, hidden_size)])
        self.hidden_layers.extend([nn.Linear(hidden_size, hidden_size) for i in range(hiddens-1)])
        self.fc_final = nn.Linear(hidden_size, obs_size)

    def forward(self, obs, a):
        x = torch.cat([obs, a], dim=-1).float()
        for fc in self.hidden_layers:
            x = F.relu(fc(x))
        delta = self.fc_final(x)
        return delta
