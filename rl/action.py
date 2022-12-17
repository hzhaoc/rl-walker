import numpy


class Action:
    def __init__(self, action: numpy.ndarray) -> None:
        self._val = action

    def val(self) -> numpy.ndarray:
        return self._val

    def shape(self) -> tuple:
        self._val.shape