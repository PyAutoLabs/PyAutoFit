from autofit.mapper.prior_model.recursion import DynamicRecursionCache


class Wrapper:
    def __init__(self, item):
        self.item = item


class A:
    def __init__(self, b=None):
        self.b = b


class B:
    def __init__(self, a=None):
        self.a = a


@DynamicRecursionCache()
def dict_recurse(item):
    try:
        for key, value in item.__dict__.items():
            setattr(item, key, dict_recurse(value))
    except AttributeError:
        pass
    return Wrapper(item)


def test_basic():
    a = A(B())
    a.b.a = a
    result = dict_recurse(a)
    assert isinstance(result.item.b.item.a, Wrapper)


def test_sub_recursion():
    a = A()
    b = A()
    c = A()

    a.b = b
    b.b = c
    c.b = b

    result = dict_recurse(a)
    assert isinstance(result, Wrapper)
