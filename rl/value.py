


class Value:
    """holds a specific reward-based value"""

    # TODO: value component
    # impelment average, discounted value
    
    def __init__(self, reward: float) -> None:
        self._reward = reward

    @property
    def reward(self) -> float:
        return self._reward