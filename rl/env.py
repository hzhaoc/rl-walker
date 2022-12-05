from __future__ import annotations
from typing import overload, Literal, Union
import gymnasium as gym


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

ENVS_USER = ["TestHumannoidEnv"]


class Env:
    """
    environment interface to create an environment for learning
    current supported environment:
    - gymnasium
    
    @prams
        id: environment id. If missing, raise error.
        kwargs: additional key-word arguments
    @return
        environment instance
    """

    @staticmethod
    @overload
    def make(id: Literal["CartPole-v0", "CartPole-v1", "MountainCar-v0", "MountainCarContinuous-v0", "Pendulum-v1", "Acrobot-v1",
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
    def make(id: Literal, **kwargs):
        if id in ENVS_GYMNASIUM:
            return gym.make(id, **kwargs)
        if id in ENVS_USER:
            from rl.test.testHumanoidEnv import TestHumannoidEnv
            return TestHumannoidEnv(**kwargs)
        raise Exception(f"{id} is not supported environment")



