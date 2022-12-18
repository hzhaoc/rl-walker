from ..util import *


class Updatable(ABC):
    """
    represent evaluation component for an updatable object
    """

    def __init__(self) -> None:
        pass

    @abstract
    def update(self) -> None:
        raise NotImplementedError

    @abstract
    def validate(self) -> None:
        raise NotImplementedError