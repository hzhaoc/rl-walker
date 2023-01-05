# TODO: remove this file

from rl.updatable.updatable import Updatable
from rl.util import *
from rl.updatable import MAX_SIZE
from rl.state import State
from rl.value import Value


class StateUpdatable(Updatable):
    """
    governs state-value function for state-evaluatable agent
    """

    def __init__(self) -> None:
        super().__init__()


class StateUpdatableFinite(Updatable):
    def __init__(self, rate = None) -> None:
        super().__init__()
        self._C = defaultdict(int)  # count of each state
        self._Q = defaultdict(float) # value of each state
        self._dim = None  # state 1-d dimension
        self._rate = rate  # if no update rate is specified, default using size of current sample

    @override
    def update(self):
        # naive incremental agent policy state value update: sample average
        try:
            self.validate()
            s = self._getState(self._agent._feedback.state)
            self._C[s] += 1
            a = self._getRate(s)
            r = self._agent._feedback.value.reward
            q = self._Q[s]
            return q + a * (r - q)
        except:
            print("cannot update state value for this agent")

    @override
    def validate(self) -> None:
        if len(self._C) > MAX_SIZE:
            raise ValueError("over {MAX_SIZE} possibel states, is this really a finite state space?")

    def _getRate(self, s: tuple) -> float:
        return self._rate if not Val.isEmptyRate(self._rate) else 1 / self._C[s]

    def _getState(self, state: State) -> tuple:
        if len(state.shape) == 1:
            return tuple(state.val)
        _s = tuple(state.flatten)
        if self._dim != None and self._dim != len(_s):
            raise ValueError("current state dimension does not match previous state dimension")
        if self._dim == None:
            self._dim = len(_s)
        return _s