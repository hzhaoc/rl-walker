from rl.updatable.updatable import Updatable
from rl.util import *
from rl.updatable import MAX_SIZE
from rl.state import State
from rl.action import Action


class ActionUpdatable(Updatable):
    """
    governs action-value function for action-evaluatable agent
    """

    def __init__(self) -> None:
        super().__init__()

    @override
    def update(self):
        return


class ActionUpdatableFinite(Updatable):
    def __init__(self, rate: float = None, 
                       action_space: dict = {},
                       discount: float = 0.95,
                ) -> None:
        super().__init__()
        self._C = defaultdict(defaultdict(float))  # outer key is state, inner key is action, value is count
        self._A = action_space
        self._dimS = None  # state 1-d dimension
        self._dimA = None  # action 1-d dimension
        self._rate = rate  # if no update rate is specified, default using size of current sample
        self._discount = discount

    @override
    def update(self):
        # naive incremental agent policy action value update: sample average
        try:
            self.validate()
            s = self._getState(self._agent._feedback.state)
            a = self._getAction(self._a)
            self._C[s][a] += 1
            alpha = self._getRate(s)
            r = self._agent._feedback.value.reward
            self._Q[s][a] = self._Q[s][a] + alpha * (r - self._Q[s][a])
        except:
            print("cannot update action value for this agent")

    @override
    def validate(self) -> None:
        if len(self._C) > MAX_SIZE:
            raise ValueError("over {MAX_SIZE} possibel states, is this really a finite state space?")

    
    def _getRate(self, s: tuple) -> float:
        return self._rate if not Val.isEmptyRate(self._rate) else 1 / self._C[s]

    def _getState(self, state: State) -> tuple:
        if len(state.shape) == 1:
            return tuple(state.val)
        s = tuple(state.flatten)
        if self._dimS != None and self._dimS != len(s):
            raise ValueError("current state dimension does not match previous state dimension")
        if self._dimS == None:
            self._dimS = len(s)
        return s

    def _getAction(self, action: Action) -> tuple:
        if len(action.shape) == 1:
            return tuple(action.val)
        a = tuple(action.flatten)
        if self._dimA != None and self._dimA != len(a):
            raise ValueError("current state dimension does not match previous state dimension")
        if self._dimA == None:
            self._dimS = len(a)
        return a