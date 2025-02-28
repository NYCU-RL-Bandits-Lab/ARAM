import torch
import torch.nn as nn
from torch.distributions import Normal
from rltorch.network import create_linear_network
import os



class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class QNetwork(BaseNetwork):
    def __init__(self, num_inputs, num_actions, num_weights, hidden_units=[256, 256],
                 initializer='xavier'):
        super(QNetwork, self).__init__()

        # https://github.com/ku2482/rltorch/blob/master/rltorch/network/builder.py
        self.Q = create_linear_network(
            num_inputs+num_actions+num_weights, num_weights, hidden_units=hidden_units,
            initializer=initializer)

    def forward(self, x):
        q = self.Q(x)
        return q

class SACQNetwork(BaseNetwork):
    def __init__(self, num_inputs, num_actions, hidden_units=[256, 256],
                 initializer='xavier'):
        super(SACQNetwork, self).__init__()

        # https://github.com/ku2482/rltorch/blob/master/rltorch/network/builder.py
        self.Q = create_linear_network(
            num_inputs+num_actions, 1, hidden_units=hidden_units,
            initializer=initializer)

    def forward(self, x):
        q = self.Q(x)
        return q

class TwinnedQNetwork(BaseNetwork):#Critic

    def __init__(self, num_inputs, num_actions, num_weights, hidden_units=[256, 256],
                 initializer='xavier'):
        super(TwinnedQNetwork, self).__init__()

        self.Q1 = QNetwork(
            num_inputs, num_actions, num_weights, hidden_units, initializer)
        self.Q2 = QNetwork(
            num_inputs, num_actions, num_weights, hidden_units, initializer)

    def forward(self, states, actions, preferences):
        x = torch.cat([states, actions ,preferences], dim=1) # preferences should be ω'
        q1 = self.Q1(x)
        q2 = self.Q2(x)
        return q1, q2

class SACTwinnedQNetwork(BaseNetwork):#Critic

    def __init__(self, num_inputs, num_actions, hidden_units=[256, 256],
                 initializer='xavier'):
        super(SACTwinnedQNetwork, self).__init__()

        self.Q1 = SACQNetwork(
            num_inputs, num_actions, hidden_units, initializer)
        self.Q2 = SACQNetwork(
            num_inputs, num_actions, hidden_units, initializer)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1) # preferences should be ω'
        q1 = self.Q1(x)
        q2 = self.Q2(x)
        return q1, q2

class GaussianPolicy(BaseNetwork):#Policy
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20
    eps = 1e-6

    def __init__(self, num_inputs, num_actions, hidden_units=[256, 256],
                 initializer='xavier'):
        super(GaussianPolicy, self).__init__()

        # https://github.com/ku2482/rltorch/blob/master/rltorch/network/builder.py
        self.policy = create_linear_network(
            num_inputs, num_actions*2, hidden_units=hidden_units,
            initializer=initializer)

    def forward(self, states, preference):

        x = torch.cat([states, preference], dim=1)
        mean, log_std = torch.chunk(self.policy(x), 2, dim=-1)
        log_std = torch.clamp(
            log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)

        return mean, log_std

    def sample(self, states, preference):
        # calculate Gaussian distribusion of (mean, std)
        means, log_stds = self.forward(states, preference)
        stds = log_stds.exp()
        normals = Normal(means, stds)
        # sample actions
        xs = normals.rsample()
        actions = torch.tanh(xs)
        # calculate entropies
        log_probs = normals.log_prob(xs)\
            - torch.log(1 - actions.pow(2) + self.eps)
        entropies = -log_probs.sum(dim=1, keepdim=True)

        return actions, entropies, torch.tanh(means)

    def sample_dist(self, states, preference):
        # calculate Gaussian distribution of (mean, std)
        means, log_stds = self.forward(states, preference)
        stds = log_stds.exp()
        
        return means, stds

    def cal_log_prob(self, states, actions, preference):
        """
        Calculate the log probability of actions given states.
        
        Args:
            states (torch.Tensor): Input states.
            actions (torch.Tensor): Actions (in [-1, 1] due to tanh).
            preference (torch.Tensor): Input preferences.
        
        Returns:
            log_probs (torch.Tensor): Log probability of the actions.
        """
        # Calculate the Gaussian distribution parameters (mean, std)
        means, log_stds = self.forward(states, preference)
        stds = log_stds.exp()
        normals = Normal(means, stds)

        # Convert actions from [-1, 1] back to untransformed space
        xs = torch.atanh(actions)  # Inverse tanh transformation

        # Compute log probability in the transformed space
        log_probs = normals.log_prob(xs) \
            - torch.log(1 - actions.pow(2) + self.eps)
        log_probs = log_probs.sum(dim=-1, keepdim=True)

        return log_probs

class SACGaussianPolicy(BaseNetwork):#Policy
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20
    eps = 1e-6
    c1 = 1/2
    c2 = 20

    def __init__(self, num_inputs, num_actions, hidden_units=[256, 256],
                 initializer='xavier'):
        super(SACGaussianPolicy, self).__init__()

        # https://github.com/ku2482/rltorch/blob/master/rltorch/network/builder.py
        self.policy = create_linear_network(
            num_inputs, num_actions*2, hidden_units=hidden_units,
            initializer=initializer)

    def forward(self, states):
        x = states
        mean, log_std = torch.chunk(self.policy(x), 2, dim=-1)
        log_std = torch.clamp(
            log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)

        return mean, log_std

    def sample(self, states):
        # calculate Gaussian distribusion of (mean, std)
        means, log_stds = self.forward(states)
        stds = log_stds.exp()
        normals = Normal(means, stds)
        # sample actions
        xs = normals.rsample()
        actions = torch.tanh(xs)

        log_probs = normals.log_prob(xs)\
            - torch.log(1 - actions.pow(2) + self.eps)

        entropies = -log_probs.sum(dim=1, keepdim=True)

        return actions, entropies, torch.tanh(means)