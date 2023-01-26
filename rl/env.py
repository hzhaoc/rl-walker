from __future__ import annotations
from typing import overload, Literal, Union, Any
import gymnasium as gym
import importlib
from rl.test.testEnv import *
import numpy as np


ENV_PKG = "rl.test.testEnv"

ENVS_GYMNASIUM = ["CartPole-v0", "CartPole-v1", "MountainCar-v0", "MountainCarContinuous-v0", "Pendulum-v1", "Acrobot-v1",
                "LunarLander-v2", "LunarLanderContinuous-v2", "BipedalWalker-v3", "BipedalWalkerHardcore-v3", "Blackjack-v1", 
                "FrozenLake-v1", "FrozenLake8x8-v1", "CliffWalking-v0", "Taxi-v3", "Reacher-v2", "Reacher-v4",
                "Pusher-v2", "Pusher-v4",
                "InvertedPendulum-v2", "InvertedPendulum-v4",
                "InvertedDoublePendulum-v2", "InvertedDoublePendulum-v4",
                "HalfCheetah-v2", "HalfCheetah-v3", "HalfCheetah-v4",
                "Hopper-v2", "Hopper-v3", "Hopper-v4",
                "Swimmer-v2", "Swimmer-v3", "Swimmer-v4",
                "Walker2d-v2", "Walker2d-v3", "Walker2d-v4",
                "Ant-v2", "Ant-v3", "Ant-v4",
                "HumanoidStandup-v2", "HumanoidStandup-v4",
                "Humanoid-v2", "Humanoid-v3", "Humanoid-v4"]
ENVS_TEST = ["TestHumannoidEnv", "TestPendulumEnv"]


class Env:
    """
    environment interface to create an environment for learning
    current supported environment:
    - gymnasium
    
    @init params
        kwargs: additional key-word arguments
    @members
        envInner:inner environment from external RL SDK such as gymnasium
    """
    
    def __init__(self, envInner: Any, **kwargs) -> None:
        self.envInner = envInner

    def step(self, action: np.ndarray) -> tuple:
        # must make sure return is Matched
        _state, _reward, _terminated, _truncated, _info = self.envInner.step(action)
        return _state, _reward, _terminated, _truncated, _info

    def reset(self, seed: int = 0) -> Union[np.ndarray, dict]:
        return self.envInner.reset(seed=seed)

    def close(self) -> None:
        self.envInner.close()

    def sample_action(self) -> np.ndarray:
        HumanoidEnv().observation_space.shape
        return self.envInner.action_space.sample()

    def sample_state(self) -> np.ndarray:
        return self.envInner.observation_space.sample()

    @property
    def shape_action(self) -> tuple:
        return self.envInner.action_space.shape

    @property
    def shape_state(self) -> tuple:
        return self.envInner.observation_space.shape

    @property
    def action_low(self) -> tuple:
        return self.envInner.action_space.low

    @property
    def action_high(self) -> tuple:
        return self.envInner.action_space.high

    @property
    def observation_low(self) -> tuple:
        return self.envInner.observation_space.low

    @property
    def observation_high(self) -> tuple:
        return self.envInner.observation_space.high

    @staticmethod
    @overload
    def make(name: Literal["CartPole-v0", "CartPole-v1", "MountainCar-v0", "MountainCarContinuous-v0", "Pendulum-v1", "Acrobot-v1",
                "LunarLander-v2", "LunarLanderContinuous-v2", "BipedalWalker-v3", "BipedalWalkerHardcore-v3", "Blackjack-v1", 
                "FrozenLake-v1", "FrozenLake8x8-v1", "CliffWalking-v0", "Taxi-v3", "Reacher-v2", "Reacher-v4",
                "Pusher-v2", "Pusher-v4",
                "InvertedPendulum-v2", "InvertedPendulum-v4",
                "InvertedDoublePendulum-v2", "InvertedDoublePendulum-v4",
                "HalfCheetah-v2", "HalfCheetah-v3", "HalfCheetah-v4",
                "Hopper-v2", "Hopper-v3", "Hopper-v4",
                "Swimmer-v2", "Swimmer-v3", "Swimmer-v4",
                "Walker2d-v2", "Walker2d-v3", "Walker2d-v4",
                "Ant-v2", "Ant-v3", "Ant-v4",
                "HumanoidStandup-v2", "HumanoidStandup-v4",
                "Humanoid-v2", "Humanoid-v3", "Humanoid-v4",
                "TestHumannoidEnv"],
                **kwargs) -> gym.Env: ...

    @staticmethod
    def make(name: Literal, **kwargs):
        if name in ENVS_GYMNASIUM:
            return Env(envInner = gym.make(name, **kwargs))
        if name in ENVS_TEST:
            try:
                module = importlib.import_module(ENV_PKG)
                cls = getattr(module, name)
                return Env(envInner = cls(**kwargs))
            except (ImportError, AttributeError) as e:
                raise ImportError(f"{ENV_PKG}.{name}")
