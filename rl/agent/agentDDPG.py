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
import math
from rl.noise import *
    

PARAMS_MIN = -1e8
PARAMS_MAX = 1e8


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
    def __init__(self, env: Env, tau: float=0.1, gamma: float=0.95, critic_lr=1e-3, actor_lr=1e-3, bufsize: int=10_000, optim_momentum: float = 1e-1,
                       actor_last_layer_weight_init: float = 3e-3, critic_last_layer_weight_init: float = 3e-4, critic_bn_eps: float = 1e-4, critic_bn_momentum: float = 1e-2,
                       actor_noise_switch=False, actor_noise_sigma=0.1, actor_noise_theta=0.1, exp_sample_size=128, actor_loss_weight_regularization_l2: float = 0.0, 
                       critic_loss_weight_regularization_l2: float = 0.0, critic_gradient_clip: float = 1e6, actor_gradient_clip: float = 1e6, update_delay=1.0, 
                       policy_noise=0.2, noise_clip=0.5) -> None:
        super().__init__()
        self.env = env
        self.critic = _CriticDDPG(input_size=self.env.shape_state[0]+self.env.shape_action[0],
                                  hidden_size=infer_size(self.env.shape_state[0]+self.env.shape_action[0]),
                                  output_size=self.env.shape_action[0],
                                  lr=critic_lr,
                                  optim_momentum=optim_momentum,
                                  last_layer_weight_init=critic_last_layer_weight_init,
                                  loss_weight_regularization_l2=critic_loss_weight_regularization_l2,
                                  gradient_clip=critic_gradient_clip,
                                  )
        self.actor = _ActorDDPG(input_size=self.env.shape_state[0], 
                                hidden_size=infer_size(self.env.shape_state[0]+self.env.shape_action[0]),
                                output_size=self.env.shape_action[0],
                                lr=actor_lr,
                                optim_momentum=optim_momentum,
                                last_layer_weight_init=actor_loss_weight_regularization_l2,
                                eps=critic_bn_eps,
                                bn_momentum=critic_bn_momentum,
                                noise=OUNoise(action_dim=self.env.shape_action[0], 
                                              low=self.env.action_low, 
                                              high=self.env.action_high, 
                                              min_sigma=actor_noise_sigma, 
                                              max_sigma=actor_noise_sigma, 
                                              theta=actor_noise_theta) if actor_noise_switch else EmptyNoise(),
                                loss_weight_regularization_l2=actor_loss_weight_regularization_l2,
                                gradient_clip=actor_gradient_clip,
                                action_low=self.env.action_low,
                                action_high=self.env.action_high,
                                )
        self.critic_target = _CriticDDPG(input_size=self.env.shape_state[0]+self.env.shape_action[0],
                                         hidden_size=infer_size(self.env.shape_state[0]+self.env.shape_action[0]),
                                         output_size=self.env.shape_action[0],
                                         lr=critic_lr,
                                         optim_momentum=optim_momentum,
                                         last_layer_weight_init=critic_last_layer_weight_init,
                                         loss_weight_regularization_l2=critic_loss_weight_regularization_l2,
                                         gradient_clip=critic_gradient_clip,
                                         )
        self.actor_target = _ActorDDPG(input_size=self.env.shape_state[0], 
                                       hidden_size=infer_size(self.env.shape_state[0]+self.env.shape_action[0]),
                                       output_size=self.env.shape_action[0],
                                       lr=actor_lr,
                                       optim_momentum=optim_momentum,
                                       last_layer_weight_init=actor_last_layer_weight_init,
                                       eps=critic_bn_eps,
                                       bn_momentum=critic_bn_momentum,
                                       loss_weight_regularization_l2=actor_loss_weight_regularization_l2,
                                       gradient_clip=actor_gradient_clip,
                                       action_low=self.env.action_low,
                                       action_high=self.env.action_high,
                                       )
        for param, param_target in zip(self.actor.parameters(), self.actor_target.parameters()):
            param_target.data.copy_(param.data)
        for param, param_target in zip(self.critic.parameters(), self.critic_target.parameters()):
            param_target.data.copy_(param.data)
        self.tau = tau   # target network update rate
        self.gamma = gamma  # future reward discount rate

        self.buf = Buffer(bufsize, exp_sample_size)
        self.noise_clip = policy_noise
        self.policy_noise = noise_clip
        
        self.train_action_low = torch.FloatTensor(np.array(self.env.action_low).reshape(1, -1).repeat(exp_sample_size, axis=0))  # affected by buf size
        self.train_action_high = torch.FloatTensor(np.array(self.env.action_high).reshape(1, -1).repeat(exp_sample_size, axis=0))  # affected by buf size

    @override(Agent)
    def act(self, state: np.ndarray) -> np.ndarray:
        return self.actor.act(torch.FloatTensor(state).reshape(1, -1))

    @override(Agent)
    def update(self) -> None:
        """
        update value function and policy

        state-action value function
        - objective is to minimize cost; use gradient update for weights

        policy
        - objective is to maximize reward; use gradient update for weights

        note:
        - update value approximator before policy approximator
        """
        if len(self.buf) <= self.buf.batch_size:
            return
        s0, a0, r0, s1, done = self.buf.sample()   # samples in batch
        s0 = torch.FloatTensor(s0)  # shape is (sample size, state space)
        a0 = torch.FloatTensor(a0)
        r0 = torch.FloatTensor(r0)
        s1 = torch.FloatTensor(s1)
        done = torch.FloatTensor(done).reshape(-1, 1) # shape is (sample size, 1)

        # 1. update online critic
        with torch.no_grad():  # q_true_biased has no gradient so it does not propogate to target network during online updating
            noise = torch.FloatTensor(a0.shape).data.normal_(0, self.policy_noise)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            a1 = self.actor_target(s1)
            a1 = (a1 + noise).clamp(self.train_action_low, self.train_action_high)
            q_true_biased = r0 + (1-done) * self.gamma * self.critic_target(s1, a1)

        q_modeled = self.critic(s0, a0)
        critic_loss = self.critic.criterion(q_modeled, q_true_biased)  # target network is detached from gradient descent
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic.gradient_clip)
        # 2. update online actor
        #   does critic need to be set at eval mode before forward? No. Critic does not have dropout or batchnorm. only these 2 are affected by eval/train
        #   note we need to freeze online critic while updating online actor
        for params in self.critic.parameters():
            params.requires_grad = False
        actor_loss = -1 * self.critic(s0, self.actor(s0)).mean()  # loss is assumed to be differentialable w.r.t. action `self.actor.forward(s0)`
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
        nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor.gradient_clip)
        for params in self.critic.parameters():
            params.requires_grad = True
        # 3. update offline (target) critic & actor
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    @override(Agent)
    def reset(self) -> None:
        self.actor.noise.reset()


class _ActorDDPG(nn.Module, Actor):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, lr: float = 3e-4, optim_momentum: float = 1e-1, last_layer_weight_init: float = 3e-3, 
                       eps: float = 1e-4, bn_momentum: float = 1e-2, noise: Noise = EmptyNoise(), loss_weight_regularization_l2: float = 0.0, gradient_clip: float = 1e6,
                       action_low: np.ndarray = np.array([]), action_high: np.ndarray = np.array([])) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        nn.init.uniform_(self.layer1.weight, -math.sqrt(1/input_size), math.sqrt(1/input_size))
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        nn.init.uniform_(self.layer2.weight, -math.sqrt(1/hidden_size), math.sqrt(1/hidden_size))
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.layer3bn = nn.BatchNorm1d(num_features=output_size, eps=eps, momentum=bn_momentum)  # NOTE 1
        nn.init.uniform_(self.layer3.weight, -last_layer_weight_init, last_layer_weight_init)
    
        #self.optimizer = optim.Adam(self.parameters(), lr=lr)  # SGD with individually-adaptive learning rate
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=1-optim_momentum, weight_decay=loss_weight_regularization_l2)  # SGD with momentum
        self.gradient_clip = gradient_clip
        self.noise = noise
        self.action_mid = torch.FloatTensor((action_low + action_high) / 2)
        self.action_radius = torch.FloatTensor((action_high - action_low) / 2)
        self.output_size = output_size

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = state
        x = relu(self.layer1(x))  # ways to alleviate vanishing gradient: relu / momental SGD / careful weight init / small learning rate / batch norm
        x = relu(self.layer2(x))
        x = tanh(self.layer3bn(self.layer3(x)))
        return self.actionScaler(x)
    
    def actionScaler(self, x: torch.Tensor) -> torch.Tensor:
        return self.action_radius * x + self.action_mid

    @override(Actor)
    def act(self, state: torch.Tensor) -> np.ndarray:
        self.eval()   # set it at eval mode. affects only batchnorm, dropout, etc
        # print(self.forward(state).detach().numpy().shape())
        with torch.no_grad():
            action = self.forward(state).detach().numpy().flatten()
        action = self.noise.get_action(action)
        self.train()  # set it back to default mode
        return action

    def _act(self, state: torch.Tensor) -> torch.Tensor:
        action = torch.FloatTensor(self.act(state)).reshape(-1, self.output_size)
        return action

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

    NOTE: batch norm in evaluation network decrease performance. 
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, lr: float = 3e-4, optim_momentum: float = 1e-1, last_layer_weight_init: float = 3e-4, 
    loss_weight_regularization_l2: float = 0.0, gradient_clip: float = 1e6) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        nn.init.uniform_(self.layer1.weight, -math.sqrt(1/input_size,), math.sqrt(1/input_size))
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        nn.init.uniform_(self.layer2.weight, -math.sqrt(1/hidden_size), math.sqrt(1/hidden_size))
        self.layer3 = nn.Linear(hidden_size, output_size)
        nn.init.uniform_(self.layer3.weight, -last_layer_weight_init, last_layer_weight_init)

        self.criterion = nn.MSELoss()
        #self.optimizer = optim.Adam(self.parameters(), lr=lr)  # SGD with individually-adaptive learning rate
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=1-optim_momentum, weight_decay=loss_weight_regularization_l2)  # SGD with momentum
        self.gradient_clip = gradient_clip

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.concat([state, action], 1)
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


def infer_size(n):
    i = 1
    while n > 1:
        n >>= 1
        i <<= 1
    i <<= 1
    i <<= 1
    return max(i, 256)

"""NOTE 1
batch norm to reduce internal covariance shift, especially when previouos layer is nonlinear. 
@see Batch Norm 2015 Paper. 
`eps` for stability of scaler post unit normalization
`momentum` for running population mean and variance. If set to None, all seen batches will be used to calcualte mean and variance
"""
