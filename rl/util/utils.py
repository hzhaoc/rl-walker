from typing import Any, Union


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

    @staticmethod
    def isEmptyRate(o: Any) -> bool:
        if isinstance(o, int) or isinstance(o, float):
            return o == None or o == 0.0 or o == 0
        if isinstance(o, list) or isinstance(o, tuple):
            return len(o) == 0 or all(Val.isEmptyRate(e) for e in o)
        if isinstance(o, dict):
            return len(o) == 0 or all(Val.isEmptyRate(e) for e in o.values())
        raise TypeError("invalid rate type")
