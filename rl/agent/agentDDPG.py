from rl.agent import Agent, Actor, Critic
from rl.util import *
from rl.env import Env
import numpy as np
import torch
from torch import sigmoid, tanh, relu
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
import torch.optim as optim
from rl.buffer import Buffer


class AgentDDPG(Agent):
    """
    Deep Deterministic Policy Gradient agent
    original idea: CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING, 2016, Timothy P. Lillicrap et al
    
    - state-action value
        - follows Bellman Optimality Equation: $q(s,a)=\sum_{s',r}\rho(s',r|s,a)(r+\gamma\max_{a'}q(s',a')))$
        - value is approximated by a non-linear approximator - neural network that inputs action and state, and outputs expected value.
    - policy
        - policy takes state and gives a determinsitic action which it expects to yield highest value
        - the policy is approximated by a different neural network that inputs state, and outputs expected optimal action.
    """
    def __init__(self, env: Env, tau: float=0.1, gamma: float=0.95, critic_lr=1e-3, actor_lr=1e-3, bufsize: int=10_000) -> None:
        super().__init__()
        # TODO: hyperparams for gamma (rewards NPV), (critic/actor) learning rate, neural network layer size
        self.env = env
        self.critic = _CriticDDPG(input_size=self.env.shape_state[0]+self.env.shape_action[0], 
                                  hidden_size=256,#self.env.shape_state[0]+self.env.shape_action[0], 
                                  output_size=self.env.shape_action[0],  # TODO: output size = action# or 1?
                                  lr=critic_lr)
        self.actor = _ActorDDPG(input_size=self.env.shape_state[0], 
                                hidden_size=256,#self.env.shape_state[0], 
                                output_size=self.env.shape_action[0],
                                lr=actor_lr)
        self.critic_target = _CriticDDPG(input_size=self.env.shape_state[0]+self.env.shape_action[0], 
                                  hidden_size=256,#self.env.shape_state[0]+self.env.shape_action[0], 
                                  output_size=self.env.shape_action[0],  # TODO: output size = action# or 1?
                                  lr=critic_lr)
        self.actor_target = _ActorDDPG(input_size=self.env.shape_state[0], 
                                hidden_size=256,#self.env.shape_state[0], 
                                output_size=self.env.shape_action[0],
                                lr=actor_lr)
        for param, param_target in zip(self.actor.parameters(), self.actor_target.parameters()):
            param_target.data.copy_(param.data)
        for param, param_target in zip(self.critic.parameters(), self.critic_target.parameters()):
            param_target.data.copy_(param.data)
        self.tau = tau   # target network update rate
        self.gamma = gamma  # future reward discount rate
        self.buf = Buffer(bufsize)

    @override(Agent)
    def act(self, state: np.ndarray) -> np.ndarray:
        return self.actor.forward(torch.FloatTensor(state)).detach().numpy()

    @override(Agent)
    def update(self, batch_size: int) -> None:
        """
        update value function and policy

        state-action value function
        - objective is to minimize cost; use gradient update for weights

        policy
        - objective is to maximize reward; use gradient update for weights

        note:
        - update value approximator before policy approximator
        """
        s0, a0, r0, s1, _ = self.buf.sample(batch_size)   # samples in batch
        s0 = torch.FloatTensor(s0)
        a0 = torch.FloatTensor(a0)
        r0 = torch.FloatTensor(r0)
        s1 = torch.FloatTensor(s1)
        # update online actor
        self.critic.optimizer.zero_grad()
        q_modeled = self.critic.forward(s0, a0)
        q_true_biased = r0 + self.gamma * self.critic_target.forward(s1, self.actor_target.forward(s1))
        critic_loss = self.critic.criterion(q_modeled, q_true_biased.detach())  # target network is deteched from gradient descent
        critic_loss.backward()
        self.critic.optimizer.step()
        # update online critic
        self.actor.optimizer.zero_grad()
        actor_loss = -1 * self.critic.forward(s0, self.actor.forward(s0)).mean()  # loss is assumed to be differentialable w.r.t. action `self.actor.forward(s0)`
        actor_loss.backward()
        self.actor.optimizer.step()
        # update offline (target) critic & actor
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


class _ActorDDPG(nn.Module, Actor):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, lr: float = 3e-4) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
    
        #self.optimizer = optim.Adam(self.parameters(), lr=lr)  # SGD with individually adaptive learning rate
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)  # SGD with momentum
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = state
        x = relu(self.layer1(x))  # ways to alliviate vanishing gradient: relu / momental SGD / careful weight init / small learning rate / batch norm
        x = relu(self.layer2(x))
        x = 2 * tanh(self.layer3(x))
        return x

    @override(Actor)
    def act(self, state: np.ndarray) -> np.ndarray:
        return self.forward(torch.FloatTensor(state)).detach().numpy()

    @override(Actor)
    def update(self, s0: torch.Tensor, critic_fwd: Any) -> None:
        raise NotImplementedError

    @override(Actor)
    def validate(self) -> None:
        return


class _CriticDDPG(nn.Module, Critic):
    """
    a non-linear approximator modeled as fully-connected neural networkthat that evaluates given action and state
    inputs action + state vector, outputs a scalar value
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, lr: float = 3e-4, gamma: float = 0.95) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

        self.criterion = nn.MSELoss()
        #self.optimizer = optim.Adam(self.parameters(), lr=lr)  # SGD with individually adaptive learning rate
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)  # SGD with momentum

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], 1)
        x = relu(self.layer1(x))
        x = relu(self.layer2(x))
        x = (self.layer3(x))
        return x

    @override(Critic)
    def update(self, s0: torch.Tensor, a0: torch.Tensor, r0: float, s1: torch.Tensor, actor_fwd: Any) -> None:
        raise NotImplementedError
        
    @override(Actor)
    def validate(self) -> None:
        return