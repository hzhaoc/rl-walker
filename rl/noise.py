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


# TODO: make exploration nosie more intelligent: explore more in worse situation; explore less in better situation
#       how to measure if a situation is getting worse or better? analyze epoch-reward long-term trend
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
    """
    def __init__(self, action_dim, low=-float("inf"),high=float("inf"), mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_dim
        # print('low', low)
        self.low          = low
        self.high         = high
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
        # action shape: (action space, ),  1d
        self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, self.t / self.decay_period)
        res = np.clip(action + self.xt, self.low, self.high)
        self.t += 1
        return res