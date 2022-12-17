import numpy


class State:
    def __init__(self, state: numpy.ndarray) -> None:
        self._val = state

    def val(self) -> numpy.ndarray:
        self._val

    def shape(self) -> tuple:
        self._val.shape