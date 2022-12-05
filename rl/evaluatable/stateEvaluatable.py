from rl.evaluatable.evaluatable import Evaluatable
from rl.util import *


class StateEvaluatable(Evaluatable):
    """
    represent state-value component for state-evaluatable agent
    """

    def __init__(self) -> None:
        super().__init__()

    @override
    def eval(self):
        # TODO: add basic incremental state value update
        # it can be sample average
        return