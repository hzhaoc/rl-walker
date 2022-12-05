from rl.evaluatable.evaluatable import Evaluatable
from rl.state import State
from rl.action import Action
from rl.util import *


class Agent:
    # TODO: complement this framework
    """
    policy agent framework

    A policy agent encapuslates:
    - policy mechanism (how actor acts on policy, how policy is updated)
    - value mechanism (how state/action value is computed, how value function is updated)

    Encapsulation comes from 2 places:
    1. class inheritence/composition
    2. clas member method/variable
    
    """

    def __init__(self, **kwargs) -> None:
        self.actor = Actor()
        self.critic = Critic()

    def act(self, S: State) -> Action:
        A = self.actor.act(S)

    def update(self, S: State, r: float) -> None:
        self.critic.eval(S, r)


class Actor(Evaluatable):
    def __init__(self) -> None:
        super().__init__()

    @abstract
    def act(self, S: State) -> Action:
        raise NotImplementedError

    @override
    @abstract
    def eval(self) -> None:
        raise NotImplementedError


class Critic(Evaluatable):
    def __init__(self) -> None:
        super().__init__()

    @override
    @abstract
    def eval(self) -> None:
        raise NotImplementedError