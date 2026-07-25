class MockClassx2:
    def __init__(self, one=1, two=2):
        self.one = one
        self.two = two


class MockClassx2Instance(MockClassx2):
    pass


class MockClassx2FormatExp:
    """
    This mock classes's second parameter `two_exp` has a format label `two={:.2e}` in `notational/label_format.ini` and
     is used to test latex generation.
    """

    def __init__(self, one=1, two_exp=2):
        self.one = one
        self.two_exp = two_exp


class MockClassx2NoSuperScript:
    def __init__(self, one=1, two=2):
        self.one = one
        self.two = two


class MockClassx4:
    def __init__(self, one=1, two=2, three=3, four=4):
        self.one = one
        self.two = two
        self.three = three
        self.four = four


class MockClassx3(MockClassx4):
    def __init__(self, one=1, two=2, three=3):
        super().__init__(one, two, three)


class MockClassx2Tuple:
    def __init__(self, one_tuple=(0.0, 0.0)):
        """Abstract MockParent, describing an object with y, x cartesian
        coordinates"""
        self.one_tuple = one_tuple

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class MockClassx3TupleFloat:
    def __init__(self, one_tuple=(0.0, 0.0), two=0.1):
        self.one_tuple = one_tuple
        self.two = two


class MockClassRelativeWidth:
    def __init__(self, one, two, three):
        self.one = one
        self.two = two
        self.three = three


class MockClassInf:
    def __init__(self, one, two):
        self.one = one
        self.two = two


class MockComplexClass:
    def __init__(self, simple: MockClassx2):
        self.simple = simple


class MockDeferredClass:
    def __init__(self, one, two):
        self.one = one
        self.two = two


class MockListClass:
    def __init__(self, ls: list):
        self.ls = ls


class MockWithFloat:
    def __init__(self, value):
        self.value = value


class MockWithTuple:
    def __init__(self, tup=(0.0, 0.0)):
        self.tup = tup


class MockOverload:
    def __init__(self, one=1.0):
        self.one = one

    def with_two(self, two):
        self.two = two

    @property
    def two(self):
        return self.one * 2

    @two.setter
    def two(self, two):
        self.one = two / 2


class MockComponents:
    def __init__(
        self,
        components_0: list = None,
        components_1: list = None,
        parameter=None,
        **kwargs
    ):
        self.parameter = parameter
        self.group_0 = components_0
        self.group_1 = components_1
        self.kwargs = kwargs


class MockParent:
    def __init__(self, tup=(0.0, 0.0)):
        self.tup = tup

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class MockChildTuple(MockParent):
    def __init__(self, tup=(0.0, 0.0)):
        """Generic circular profiles class to contain functions shared by light and
        mass profiles.

        Parameters
        ----------
        tup
            The (y,x) coordinates of the origin of the profile.
        """
        super().__init__(tup)


class MockChildTuplex2(MockChildTuple):
    def __init__(self, tup=(0.0, 0.0), one=1.0, two=0.0):
        """Generic elliptical profiles class to contain functions shared by light
        and mass profiles.

        Parameters
        ----------
        tup
            The (y,x) coordinates of the origin of the profiles
        one
            Ratio of profiles ellipse's minor and major axes (b/a)
        two : float
            Rotational two of profiles ellipse counter-clockwise from positive x-axis
        """
        super().__init__(tup)
        self.one = one
        self.two = two


class MockChildTuplex3(MockChildTuple):
    def __init__(self, tup=(0.0, 0.0), one=1.0, two=0.0, three=0.0):
        """Generic elliptical profiles class to contain functions shared by light
        and mass profiles.

        Parameters
        ----------
        tup
            The (y,x) coordinates of the origin of the profiles
        one
            Ratio of profiles ellipse's minor and major axes (b/a)
        two : float
            Rotational two of profiles ellipse counter-clockwise from positive x-axis
        """
        super().__init__(tup)
        self.one = one
        self.two = two
        self.three = three


class Parameter:
    def __init__(self, value: float = 0.5):
        self.value = value


class WithString:
    def __init__(self, arg: "Parameter"):
        self.arg = arg


class WithConstants:
    def __init__(self, constant_1, constant_2):
        self.constant_1 = constant_1
        self.constant_2 = constant_2


class ModelWithTupleConstant:
    def __init__(self, constant=(0.0, 0.0)):
        self.constant = constant
