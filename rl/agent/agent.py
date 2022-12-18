from rl.updatable.updatable import Updatable
from rl.state import State
from rl.action import Action
from rl.util import *
from rl.env import EnvFeedback


class Agent(ABC):
    # TODO: complete this framework
    """
    policy agent framework

    A policy agent encapuslates:
    - policy mechanism (how actor acts on policy, how policy is updated)
    - value mechanism (how state/action value is computed, how value function is updated)

    Encapsulation comes from 2 places:
    1. class inheritence/composition
    2. class member method/variable
    
    """

    def __init__(self, **kwargs) -> None:
        self.actor = Actor(self)
        self.critic = Critic(self)
        self._feedback = EnvFeedback()  # current feedback
        self._a = Action() # current action

    def act(self, S: State) -> Action:
        return self.actor.act(S)

    def update(self, feedback: EnvFeedback) -> None:
        self.critic.update(feedback)  # value function should be updated first to provde latest estimation adjustment to actor
        self.actor.update(feedback)


class Actor(Updatable, ABC):
    """
    actor governs agent's policy
    specifically, what actions to take based on environment feedback such as reward, state, etc,
    and how to learn to maximzie long term rewrd thru this state-action-state-... interaction
    """
    def __init__(self, agent: Agent) -> None:
        super().__init__()
        self._agent = agent

    @abstract
    def act(self, S: State) -> Action:
        """take what the action thinks as the best action from feedback"""
        raise NotImplementedError

    @override
    @abstract
    def update(self) -> None:
        """update current policy"""
        raise NotImplementedError


class Critic(Updatable, ABC):
    def __init__(self, agent: Agent) -> None:
        super().__init__()
        self._agent = agent

    @override
    @abstract
    def update(self) -> None:
        """updte current agent current value function"""
        raise NotImplementedError