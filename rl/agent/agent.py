from rl.updatable.updatable import Updatable
from rl.util import *
import numpy as np


class Agent(ABC):
    """
    policy agent framework

    A policy agent encapuslates:
    - policy mechanism (how actor acts on policy, how policy is updated)
    - value mechanism (how state/action value is computed, how value function is updated)

    Encapsulation comes from 2 places:
    1. class inheritence/composition
    2. class member method/variable
    
    """

    def __init__(self) -> None:
        return

    @abstract
    def act(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstract
    def update(self) -> None:
        raise NotImplementedError

    @abstract
    def reset(self) -> None:
        raise NotImplementedError


class Actor(Updatable, ABC):
    """
    actor governs agent's policy
    specifically, what actions to take based on environment feedback such as reward, state, etc,
    and how to learn to maximzie long term rewrd thru this state-action-state-... interaction
    """
    def __init__(self) -> None:
        super().__init__()

    @abstract
    def act(self, s: np.ndarray) -> np.ndarray:
        """take what the action thinks as the best action from feedback"""
        raise NotImplementedError

    @override
    @abstract
    def update(self) -> None:
        """update current policy"""
        raise NotImplementedError


class Critic(Updatable, ABC):
    def __init__(self) -> None:
        super().__init__()

    @override
    @abstract
    def update(self) -> None:
        """updte current agent current value function"""
        raise NotImplementedError