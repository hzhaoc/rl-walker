# TODO: remove this file

import numpy


class State:
    def __init__(self, state: numpy.ndarray) -> None:
        self._val = state

    @property
    def val(self) -> numpy.ndarray:
        self._val

    @property
    def shape(self) -> tuple:
        self._val.shape

    @property
    def flatten(self) -> numpy.ndarray:
        self._val.flatten()