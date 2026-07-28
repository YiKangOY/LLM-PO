# Author: anonymous


import torch
import torch.nn as nn
from torch.nn import init


class MLPBase(nn.Module):
    def __init__(self, observation_shape, action_shape, reward_shape):
        super(MLPBase, self).__init__()
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.reward_shape = reward_shape
        self.base = nn.Sequential(
            nn.Linear(self.observation_shape + self.reward_shape, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        return self.base(x)


class BOOMActorCriticNetwork(MLPBase):
    def __init__(self, observation_shape, action_shape, reward_shape):
        super(BOOMActorCriticNetwork, self).__init__(
            observation_shape,
            action_shape,
            reward_shape
        )
        self.actor = nn.Sequential(
            nn.Linear(self.observation_shape + self.reward_shape, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.action_shape)

        )
        self.critic = nn.Sequential(
            nn.Linear(self.observation_shape + self.reward_shape, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.reward_shape)
        )
        self.init_network()

    def init_network(self):
        for p in self.modules():
            if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()
    
    def forward(self, state, preference):
        x = torch.cat((state, preference), dim=1)
        # x = self.base(x)
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value


class RocketActorCriticNetwork(MLPBase):
    def __init__(self, observation_shape, action_shape, reward_shape):
        super(RocketActorCriticNetwork, self).__init__(
            observation_shape,
            action_shape,
            reward_shape
        )
        self.actor = nn.Sequential(
            nn.Linear(self.observation_shape + self.reward_shape, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.action_shape)
        )
        self.critic = nn.Sequential(
            nn.Linear(self.observation_shape + self.reward_shape, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.reward_shape)
        )
        self.init_network()

    def init_network(self):
        for p in self.modules():
            if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

    def forward(self, state, preference):
        x = torch.cat((state, preference), dim=1)
        # x = self.base(state)
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value


class PPOActorCriticNetwork(BOOMActorCriticNetwork):
    """
        Alias of BOOMActorCriticNetwork to keep a stable interface for
        policy/critic wiring.
    """
    pass


class ECOActorCriticNetwork(MLPBase):
    """
        ECO variant of the dual-head actor-critic used across the reference
        framework. The observation/action/reward shapes are provided at
        construction so the same network can be reused in the ECO agent.
    """
    def __init__(self, observation_shape, action_shape, reward_shape):
        super(ECOActorCriticNetwork, self).__init__(
            observation_shape,
            action_shape,
            reward_shape
        )
        self.actor = nn.Sequential(
            nn.Linear(self.observation_shape + self.reward_shape, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.action_shape)
        )
        self.critic = nn.Sequential(
            nn.Linear(self.observation_shape + self.reward_shape, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.reward_shape)
        )
        self.init_network()

    def init_network(self):
        for p in self.modules():
            if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

    def forward(self, state, preference):
        x = torch.cat((state, preference), dim=1)
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value
