from functools import wraps


def cast_collection(named_tuple):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return list(map(lambda tup: named_tuple(*tup), func(*args, **kwargs)))

        return wrapper

    return decorator


class AttributeNameValue:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __iter__(self):
        return iter(self.tuple)

    @property
    def tuple(self):
        return self.name, self.value

    def __getitem__(self, item):
        return self.tuple[item]

    def __eq__(self, other):
        if isinstance(other, AttributeNameValue):
            return self.tuple == other.tuple
        if isinstance(other, tuple):
            return self.tuple == other
        return False

    def __hash__(self):
        return hash(self.tuple)

    def __str__(self):
        return "({}, {})".format(self.name, self.value)

    def __repr__(self):
        return "<{} {}>".format(self.__class__.__name__, str(self))


class PriorNameValue(AttributeNameValue):
    @property
    def prior(self):
        return self.value


class InstanceNameValue(AttributeNameValue):
    @property
    def instance(self):
        return self.value


class DeferredNameValue(AttributeNameValue):
    @property
    def deferred(self):
        return self.value
