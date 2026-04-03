import itertools
import random
from abc import ABC, abstractmethod
from copy import copy
from typing import Union, Tuple, Optional, Dict

from autoconf import conf

from autofit.mapper.prior.arithmetic import ArithmeticMixin
from autofit.mapper.prior.constant import Constant
from autofit.mapper.prior.deferred import DeferredArgument
from autofit.mapper.variable import Variable

epsilon = 1e-14


class Prior(Variable, ABC, ArithmeticMixin):
    __database_args__ = ("id_")

    _ids = itertools.count()

    def __init__(self, message, id_=None):
        """
        An object used to mappers a unit value to an attribute value for a specific
        class attribute.

        Parameters
        ----------
        message

        """
        super().__init__(id_=id_)
        self.message = message
        message.id_ = self.id

        self.width_modifier = None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Create a prior from a flattened PyTree

        Parameters
        ----------
        aux_data
            Auxiliary information that remains unchanged including
            the keys of the dict
        children
            Child objects subject to change

        Returns
        -------
        An instance of this class
        """
        return cls(*children)

    def unit_value_for(self, physical_value: float) -> float:
        """
        Compute the unit value between 0 and 1 for the physical value.
        """
        return self.message.cdf(physical_value)

    def with_message(self, message):
        """Return a copy of this prior with a different message (distribution).

        Parameters
        ----------
        message
            The new message object defining the prior's distribution.

        Returns
        -------
        Prior
            A copy of this prior using the new message.
        """
        new = copy(self)
        new.message = message
        return new

    def new(self):
        """
        Returns a copy of this prior with a new id assigned making it distinct
        """
        new = copy(self)
        new.id = next(self._ids)
        return new

    @abstractmethod
    def with_limits(self, lower_limit: float, upper_limit: float) -> "Prior":
        """
        Create a new instance of the same prior class with the passed limits.
        """

    @property
    def factor(self):
        """
        A callable PDF used as a factor in factor graphs
        """
        return self.message.factor

    @staticmethod
    def for_class_and_attribute_name(cls, attribute_name):
        """Create a prior from the configuration for a given class and attribute.

        Looks up the prior type and parameters in the prior config files
        for the specified class and attribute name.

        Parameters
        ----------
        cls
            The model class whose config is looked up.
        attribute_name
            The name of the attribute on that class.

        Returns
        -------
        Prior
            A prior instance constructed from the config entry.
        """
        prior_dict = conf.instance.prior_config.for_class_and_suffix_path(
            cls, [attribute_name]
        )
        return Prior.from_dict(prior_dict)

    def random(
        self,
        lower_limit : float = 0.0,
        upper_limit : float = 1.0
    ) -> float:
        """
        A random value sampled from this prior
        """
        return self.value_for(
            random.uniform(
                lower_limit,
                upper_limit,
            )
        )

    def value_for(self, unit: float) -> float:
        """
        Return a physical value for a value between 0 and 1 with the transformation
        described by this prior.

        Parameters
        ----------
        unit
            A unit value between 0 and 1.

        Returns
        -------
        A physical value, mapped from the unit value accoridng to the prior.
        """
        return self.message.value_for(unit)

    def instance_for_arguments(
        self,
        arguments,
        ignore_assertions=False,
    ):
        """Look up this prior's value in an arguments dictionary.

        Parameters
        ----------
        arguments
            A dictionary mapping Prior objects to physical values.
        ignore_assertions
            Unused for priors (present for interface compatibility).
        """
        _ = ignore_assertions
        return arguments[self]

    def project(self, samples, weights):
        """Project this prior given samples and log weights from a search.

        Returns a copy of this prior whose message has been updated to
        reflect the posterior information from the samples.

        Parameters
        ----------
        samples
            Array of sample values for this parameter.
        weights
            Log weights for each sample.
        """
        result = copy(self)
        result.message = self.message.project(
            samples=samples,
            log_weight_list=weights,
            id_=self.id,
        )
        return result

    def __getattr__(self, item):
        if item in ("__setstate__", "__getstate__"):
            raise AttributeError(item)
        return getattr(self.message, item)

    def __eq__(self, other):
        try:
            return self.id == other.id
        except AttributeError:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return "<{} id={}>".format(
            self.__class__.__name__, self.id,
        )

    def __str__(self):
        """The line of text describing this prior for the model_mapper.info file"""
        return f"{self.__class__.__name__} [{self.id}], {self.parameter_string}"

    @property
    @abstractmethod
    def parameter_string(self) -> str:
        """A human-readable string summarizing this prior's parameters.

        Subclasses must implement this to return a description such as
        ``"mean = 0.0, sigma = 1.0"`` or ``"lower_limit = 0.0, upper_limit = 1.0"``.
        """
        pass

    def __float__(self):
        return self.value_for(0.5)

    @classmethod
    def from_dict(
        cls,
        prior_dict: dict,
        reference: Optional[Dict[str, str]] = None,
        loaded_ids: Optional[Dict[int, "Prior"]] = None,
    ) -> Union["Prior", DeferredArgument]:
        """
        Returns a prior from a JSON representation.

        Parameters
        ----------
        prior_dict : dict
            A dictionary representation of a prior including a type (e.g. Uniform) and all constructor arguments.
        reference
            A dictionary mapping prior ids to the priors they reference.
        loaded_ids
            A dictionary mapping prior ids to the priors they have loaded.

        Returns
        -------
        An instance of a child of this class.
        """
        loaded_ids = {} if loaded_ids is None else loaded_ids
        id_ = prior_dict.get("id")
        try:
            return loaded_ids[id_]
        except KeyError:
            pass
        if prior_dict["type"] == "Deferred":
            return DeferredArgument()

        from .uniform import UniformPrior
        from .log_uniform import LogUniformPrior
        from .gaussian import GaussianPrior
        from .log_gaussian import LogGaussianPrior
        from .truncated_gaussian import TruncatedGaussianPrior

        prior_type_dict = {
            "Uniform": UniformPrior,
            "LogUniform": LogUniformPrior,
            "Gaussian": GaussianPrior,
            "LogGaussian": LogGaussianPrior,
            "TruncatedGaussian" : TruncatedGaussianPrior,
            "Constant": Constant,
        }

        # noinspection PyProtectedMember
        prior = prior_type_dict[prior_dict["type"]](
            **{
                key: value
                for key, value in prior_dict.items()
                if key not in ("type", "width_modifier", "limits", "id")
            },
        )
        if id_ is not None:
            loaded_ids[id_] = prior

        return prior

    def dict(self) -> dict:
        """
        A dictionary representation of this prior
        """
        prior_dict = {
            "type": self.name_of_class(),
            "id": self.id,
        }
        return prior_dict

    @classmethod
    def name_of_class(cls) -> str:
        """
        A string name for the class, with the prior suffix removed.
        """
        return cls.__name__.replace("Prior", "")

    @property
    def limits(self) -> Tuple[float, float]:
        """The (lower, upper) bounds of this prior.

        Returns (-inf, inf) by default. Subclasses with finite bounds
        (e.g. UniformPrior) override this.
        """
        return (float("-inf"), float("inf"))

    def gaussian_prior_model_for_arguments(self, arguments):
        """Look up this prior in an arguments dict and return the mapped value.

        Used during prior replacement workflows where each prior is mapped
        to a new prior or fixed value via an arguments dictionary.

        Parameters
        ----------
        arguments
            A dictionary mapping Prior objects to their replacement values.
        """
        return arguments[self]
