from typing import Any, Union
from collections import deque


def override(interface_class):
    def overrider(method):
        assert(method.__name__ in dir(interface_class))
        return method
    return overrider


class Val:
    @staticmethod
    def isEmpty(o: Any) -> bool:
        if isinstance(o, str):
            return o == ""
        if isinstance(o, list) or isinstance(o, tuple):
            return len(o) == 0 or all(Val.isEmpty(e) for e in o)
        if isinstance(o, dict):
            return len(o) == 0 or all(Val.isEmpty(e) for e in o.values())
        return o == None


class Queue:
    def __init__(self, maxlen=100) -> None:
        self.l = maxlen
        self.q = deque(maxlen=self.l)
        self.s = 0

    def append(self, e):
        if (len(self.q) == self.l):
            self.s -= self.q[0]
        self.s += e
        self.q.append(e)

    @property
    def sum(self):
        return self.s

    def __len__(self):
        return self.q.__len__()

    @property
    def len(self):
        return self.__len__()

    @property
    def avg(self):
        return self.sum / self.len
