class ArithmeticMixin:
    def __add__(self, other):
        """
        Add this object to another object. Addition occurs
        after priors have been converted into values.

        Parameters
        ----------
        other

        Returns
        -------
        An object comprising two objects to be summed after
        realisation
        """
        from autofit.mapper.prior.arithmetic.compound import SumPrior
        return SumPrior(
            self, other
        )

    def __floordiv__(self, other):
        from autofit.mapper.prior.arithmetic.compound import FloorDivPrior
        return FloorDivPrior(self, other)

    def __rfloordiv__(self, other):
        from autofit.mapper.prior.arithmetic.compound import FloorDivPrior
        return FloorDivPrior(other, self)

    def __abs__(self):
        from autofit.mapper.prior.arithmetic.compound import AbsolutePrior
        return AbsolutePrior(self)

    def __truediv__(self, other):
        from autofit.mapper.prior.arithmetic.compound import DivisionPrior
        return DivisionPrior(self, other)

    def __rtruediv__(self, other):
        from autofit.mapper.prior.arithmetic.compound import DivisionPrior
        return DivisionPrior(other, self)

    def __pow__(self, other):
        from autofit.mapper.prior.arithmetic.compound import PowerPrior
        return PowerPrior(self, other)

    def __rpow__(self, other):
        from autofit.mapper.prior.arithmetic.compound import PowerPrior
        return PowerPrior(other, self)

    def __mod__(self, other):
        from autofit.mapper.prior.arithmetic.compound import ModPrior
        return ModPrior(self, other)

    def __rmod__(self, other):
        from autofit.mapper.prior.arithmetic.compound import ModPrior
        return ModPrior(other, self)

    def __radd__(self, other):
        """
        Add this object to another object. Addition occurs
        after priors have been converted into values.

        Parameters
        ----------
        other

        Returns
        -------
        An object comprising two objects to be summed after
        realisation
        """
        from autofit.mapper.prior.arithmetic.compound import SumPrior
        return SumPrior(
            other, self
        )

    def __sub__(self, other):
        """
        Subtract another object from this object. Subtraction
        occurs after priors have been converted into values.

        Parameters
        ----------
        other

        Returns
        -------
        An object comprising two objects to be summed after
        realisation
        """
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __neg__(self):
        """
        Returns an object representing the negation of this
        object.
        """
        from autofit.mapper.prior.arithmetic.compound import NegativePrior
        return NegativePrior(
            self
        )

    def __mul__(self, other):
        """
        Multiple another object by this object. Multiplication
        occurs after priors have been converted into values.

        Parameters
        ----------
        other

        Returns
        -------
        An object comprising two objects to be multiplied after
        realisation
        """
        from autofit.mapper.prior.arithmetic.compound import MultiplePrior
        return MultiplePrior(
            self, other
        )

    def __rmul__(self, other):
        from autofit.mapper.prior.arithmetic.compound import MultiplePrior
        return MultiplePrior(
            other, self
        )

    def __gt__(self, other_prior):
        """
        Add an assertion that values associated with this prior are greater.

        Parameters
        ----------
        other_prior
            Another prior which is associated with a field that should always have
            lower physical values.

        Returns
        -------
        An assertion object
        """
        from autofit.mapper.prior.arithmetic.assertion import GreaterThanLessThanAssertion, unwrap
        # noinspection PyTypeChecker
        return GreaterThanLessThanAssertion(
            greater=unwrap(self),
            lower=unwrap(other_prior)
        )

    def __lt__(self, other_prior):
        """
        Add an assertion that values associated with this prior are lower.

        Parameters
        ----------
        other_prior
            Another prior which is associated with a field that should always have
            greater physical values.

        Returns
        -------
        An assertion object
        """
        from autofit.mapper.prior.arithmetic.assertion import GreaterThanLessThanAssertion, unwrap
        # noinspection PyTypeChecker
        return GreaterThanLessThanAssertion(
            lower=unwrap(self),
            greater=unwrap(other_prior)
        )

    def __ge__(self, other_prior):
        """
        Add an assertion that values associated with this prior are greater or equal.

        Parameters
        ----------
        other_prior
            Another prior which is associated with a field that should always have
            lower physical values.

        Returns
        -------
        An assertion object
        """
        from autofit.mapper.prior.arithmetic.assertion import GreaterThanLessThanEqualAssertion, unwrap
        # noinspection PyTypeChecker
        return GreaterThanLessThanEqualAssertion(
            greater=unwrap(self),
            lower=unwrap(other_prior)
        )

    def __le__(self, other_prior):
        """
        Add an assertion that values associated with this prior are lower or equal.

        Parameters
        ----------
        other_prior
            Another prior which is associated with a field that should always have
            greater physical values.

        Returns
        -------
        An assertion object
        """
        from autofit.mapper.prior.arithmetic.assertion import GreaterThanLessThanEqualAssertion, unwrap
        # noinspection PyTypeChecker
        return GreaterThanLessThanEqualAssertion(
            lower=unwrap(self),
            greater=unwrap(other_prior)
        )
