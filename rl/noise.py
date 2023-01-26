"""noise.py: stochastic process simulation"""

__author__      = "rail-berkeley"


import numpy as np
from rl.util import *


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


class OUNoise(Noise):
    """
    Ornstein-Ulhenbeck Process
    https://github.com/rail-berkeley/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
    """
    def __init__(self, action_dim, low=-float("inf"),high=float("inf"), mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_dim
        self.low          = low
        self.high         = high
        self.reset()
    
    @override(Noise)
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        self.t = 0
        
    @override(Noise)
    def evolve_state(self) -> np.ndarray:
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    @override(Noise)
    def get_action(self, action: np.ndarray) -> np.ndarray:
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, self.t / self.decay_period)
        res = np.clip(action + ou_state, self.low, self.high)
        self.t += 1
        return res