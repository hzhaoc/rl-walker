import numpy as np
from rl.util import *
from params import Params
from rl.env import Env


class Noise:
    @abstract
    def reset(self):
        raise NotImplementedError
    
    @abstract
    def evolve_state(self) -> None:
        raise NotImplementedError
    
    @abstract
    def get_action(self, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def make(params: Params, env: Env):
        if params.agent.noise_type == 'ou':
            print(f"use ou noise of sigma {params.agent.actor_noise_sigma}, theta {params.agent.actor_noise_theta}")
            return OUNoise(action_dim=env.shape_action[0],
                            low=env.action_low,
                            high=env.action_high,
                            min_sigma=params.agent.actor_noise_sigma,
                            max_sigma=params.agent.actor_noise_sigma,
                            theta=params.agent.actor_noise_theta)
        if params.agent.noise_type == 'normal':
            print(f'use normal noise of sigma {params.agent.actor_noise_sigma}')
            return NormalNoise(sigma=params.agent.actor_noise_sigma, 
                                dim=env.shape_action[0],
                                low=env.action_low,
                                high=env.action_high)
        if params.agent.noise_type == 'empty':
            print('use empty noise')
            return EmptyNoise()
        raise ValueError(f"noise type {params.agent.type} is not supported")


class EmptyNoise(Noise):
    def __init__(self):
        self.reset()
    
    @override(Noise)
    def reset(self):
        pass
    
    @override(Noise)
    def evolve_state(self) -> None:
        pass
    
    @override(Noise)
    def get_action(self, action: np.ndarray) -> np.ndarray:
        return action


class NormalNoise(Noise):
    '''noise of normal distribution with mu of 0, sigma from user input'''
    def __init__(self, sigma=1, dim=1, low=-float("inf"),high=float("inf")):
        self.reset()
        self.sigma = sigma
        self.dim = dim # action dimension
        self.low = low # clip action value low
        self.high = high # clip action value high
    
    @override(Noise)
    def reset(self):
        pass
    
    @override(Noise)
    def evolve_state(self) -> None:
        pass
    
    @override(Noise)
    def get_action(self, action: np.ndarray) -> np.ndarray:
        # input and output shape: (k, ),  1d
        action = action + np.random.normal(0, self.sigma, size=self.dim)
        action = action.clip(self.low, self.high)
        return action


class OUNoise(Noise):
    """
    Ornstein-Ulhenbeck Process
    https://github.com/rail-berkeley/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py

    This strategy implements the Ornstein-Uhlenbeck process, which adds
    time-correlated noise to the actions taken by the deterministic policy.
    The OU process satisfies the following stochastic differential equation:
    dxt = theta*(mu - xt)*dt + sigma*dWt
    where Wt denotes the Wiener process

    Based on the rllab implementation.
    author: rail-berkeley
    """
    def __init__(self, action_dim, low=-float("inf"),high=float("inf"), mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_dim
        self.low          = low # clip low
        self.high         = high # clip high
        self.reset()
    
    @override(Noise)
    def reset(self):
        self.xt = np.ones(self.action_dim) * self.mu
        self.t = 0
        
    @override(Noise)
    def evolve_state(self) -> np.ndarray:
        dxt = self.theta * (self.mu - self.xt) + self.sigma * np.random.randn(self.action_dim)
        self.xt += dxt
    
    @override(Noise)
    def get_action(self, action: np.ndarray) -> np.ndarray:
        # input and output shape: (k, ),  1d
        self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, self.t / self.decay_period)
        res = np.clip(action + self.xt, self.low, self.high)
        self.t += 1
        return res