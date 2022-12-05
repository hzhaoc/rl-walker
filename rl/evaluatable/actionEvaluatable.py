from rl.evaluatable.evaluatable import Evaluatable
from rl.util import *


class ActionEvaluatable(Evaluatable):
    """
    represent action-value component for action-evaluatable agent
    """

    def __init__(self) -> None:
        super().__init__()

    @override
    def eval(self):
        # TODO: add basic incremental update on action value
        # it can be sample average
        # 
        return