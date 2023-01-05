from rl.agent import Agent, Actor, Critic
from rl.util import *
from rl.env import EnvFeedback, Env
import numpy as np
import torch
from torch import sigmoid, tanh
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
import torch.optim as optim


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
    def __init__(self, env: Env) -> None:
        super().__init__()
        # TODO: hyperparams for gamma (rewards NPV), (critic/actor) learning rate, neural network layer size
        self.env = env
        self.critic = _CriticDDPG(input_size=self.env.shape_state[0]+self.env.shape_action[0], 
                                  hidden_size=(self.env.shape_state[0]+self.env.shape_action[0]) // 2, 
                                  output_size=1, 
                                  lr=1e-3, 
                                  gamma=0.95)
        self.actor = _ActorDDPG(input_size=self.env.shape_state[0], 
                                hidden_size=self.env.shape_state[0] // 2, 
                                output_size=self.env.shape_action[0],
                                lr=1e-5)

    @override(Agent)
    def act(self, state: np.ndarray) -> np.ndarray:
        self.s0 = torch.from_numpy(state)
        self.a0 = self.actor.forward(self.s0)
        return self.a0.detach().numpy()

    @override(Agent)
    def update(self, feedback: EnvFeedback) -> None:
        """
        update value function and policy

        state-action value function
        - objective is to minimize cost; use gradient update for weights

        policy
        - objective is to maximize reward; use gradient update for weights

        note:
        - update value approximator before policy approximator
        """

        self.r0 = feedback.reward
        self.s1 = torch.from_numpy(feedback.state)

        self.critic.update(self.s0, self.a0, self.r0, self.s1, self.actor.forward)
        self.actor.update(self.s0, self.critic.forward)


class _ActorDDPG(nn.Module, Actor):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, lr: float = 3e-4) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
    
        self.optimizer = optim.Adam(self.parameters(), lr=lr)  # a SGD with momentum-adaptive learning rate
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = sigmoid(self.layer1(state.float()))
        x = sigmoid(self.layer2(x))
        #x = sigmoid(self.layer3(x)) # TODO: de-nomalize action space back to "actuator"
        x = 0.4 * torch.tanh(self.layer3(x))
        return x

    @override(Actor)
    def act(self, state: np.ndarray) -> np.ndarray:
        return self.forward(torch.from_numpy(state)).detach().numpy()

    @override(Actor)
    def update(self, s0: torch.Tensor, critic_fwd: Any) -> None:
        # (-1 * critic_fwd(s0, self.forward(s0))).backward()
        (-1 * critic_fwd(s0, self.forward(s0))).mean().backward()  # assuming critic_fwd is differentialable w.r.t. action 'a = self.forward(s0)'
        self.optimizer.step()
        self.optimizer.zero_grad()

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
        self.optimizer = optim.Adam(self.parameters(), lr=lr)  # a SGD with momentum-adaptive learning rate
        self.gamma = gamma

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], 0)
        x = sigmoid(self.layer1(x.float()))
        x = sigmoid(self.layer2(x))
        #x = sigmoid(self.layer3(x))  # better without normalizing output?
        x = self.layer3(x)
        return x

    @override(Critic)
    def update(self, s0: torch.Tensor, a0: torch.Tensor, r0: float, s1: torch.Tensor, actor_fwd: Any) -> None:
        q_modeled = self.forward(s0, a0)
        # TODO: add an option to use differential return rather than NPV
        q_true_biased = r0 + self.gamma * self.forward(s1, actor_fwd(s1))
        self.criterion(q_modeled, q_true_biased).backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    @override(Critic)
    def validate(self) -> None:
        return