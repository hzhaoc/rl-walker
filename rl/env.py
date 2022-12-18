from __future__ import annotations
from typing import overload, Literal, Union, Any
import gymnasium as gym
import importlib
from rl.test.testEnv import *
import numpy
import copy
from rl.value import Value
from rl.state import State
from rl.action import Action


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
ENVS_TEST = ["TestHumannoidEnv"]


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

    def step(self, action: Action) -> EnvFeedback:
        _state, _reward, _terminated, _truncated, _info = self.envInner.step(action.val())  # must make sure return is Matched
        return EnvFeedback(state = State(state=_state),
                           value = Value(reward=_reward),
                           terminated = _terminated,
                           truncated = _truncated,
                           info = _info)

    def reset(self, seed: int = 0) -> Union[State, dict]:
        _state, _info = self.envInner.reset(seed=seed)
        return State(state=_state), _info

    def close(self) -> None:
        self.envInner.close()

    def sampleAction(self) -> Action:
        return Action(self.envInner.action_space.sample())

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


class EnvFeedback:
    """
    basically like a C++ struct that holds a group of variable values
    which together forms an envrionment feedback to agent
    """

    def __init__(self, state: State = None,
                       value: Value = None,
                       terminated: bool = None,
                       truncated: bool = None,
                       info: dict = None) -> None:
        self.state = state
        self.value = value
        self.terminated = terminated
        self.truncated = truncated
        self.info = copy.deepcopy(info)

    def isEmpty(self) -> bool:
        return (self.state == None) \
            or (self.value == None) \
            or (self.terminated == None) \
            or (self.truncated == None) \
            or (self.info == None)


