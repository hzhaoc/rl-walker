import numpy


class Action:
    def __init__(self, action: numpy.ndarray) -> None:
        self._val = action

    @property
    def val(self) -> numpy.ndarray:
        self._val

    @property
    def shape(self) -> tuple:
        self._val.shape

    @property
    def flatten(self) -> numpy.ndarray:
        self._val.flatten()