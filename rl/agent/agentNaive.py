from rl.agent import Agent, Actor, Critic
from rl.updatable.actionUpdatable import ActionUpdatableFinite
from rl.state import State
from rl.action import Action
from rl.util import *


class AgentNaive(Agent):
    """
    navie agent
    policy: take action that maixmizes action value from finite state-action space
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._feedback = None  # current feedback
        self.actor = ActorNaive(self)
        self.critic = CriticNaive(self)


class ActorNaive(Actor, ActionUpdatableFinite):
    def __init__(self, agent: Agent, 
                       action_space: dict = {}) -> None:
        Actor.__init__(self, agent = agent)
        ActionUpdatableFinite.__init__(self, action_space = action_space)
        self._Q = defaultdict(defaultdict(float)) # outer key is state, inner key is action, value is value
        self._a = None

    @override
    def act(self, S: State) -> Action:
        self.validate()
        s = self._getState(s)
        self._a = Action(action=max(self._Q[s], key = self._Q[s].get))
        return self._a

    @override
    def update(self) -> None:
        raise NotImplementedError


class CriticNaive(Critic, ActionUpdatableFinite):
    def __init__(self, agent: Agent, 
                       rate: float = None,
                       action_space: dict = {}) -> None:
        Critic.__init__(self, agent = agent)
        ActionUpdatableFinite.__init__(self, rate = rate, action_space = action_space)

    @override
    def update(self):
        ActionUpdatableFinite.update(self)