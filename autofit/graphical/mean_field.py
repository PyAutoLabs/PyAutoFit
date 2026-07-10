import logging
from collections import ChainMap
from numbers import Real
from typing import Dict, Tuple, Optional, Union, Iterable
from operator import attrgetter

import numpy as np

from autoconf import cached_property
from autofit import exc
from autofit.graphical.factor_graphs.abstract import AbstractNode
from autofit.graphical.factor_graphs.factor import Factor
from autofit.graphical.factor_graphs.jacobians import AbstractJacobian
from autofit.graphical.utils import (
    StatusFlag,
    prod,
    OptResult,
    Status,
    LogWarnings,
)
from autofit.mapper.prior.abstract import Prior
from autofit.mapper.prior_model.collection import Collection
from autofit.mapper.variable import (
    Variable,
    Plate,
    VariableData,
    FactorValue,
)
from autofit.mapper.variable_operator import VariableFullOperator
from autofit.messages.abstract import AbstractMessage
from autofit.messages.fixed import FixedMessage
from autofit.messages.interface import MessageInterface

VariableFactorDist = Dict[str, Dict[Factor, AbstractMessage]]
Projection = Dict[str, AbstractMessage]

logger = logging.getLogger(__name__)

_log_projection_warnings = logger.debug

Delta = Union[float, "MeanField"]


def is_message(message):
    return isinstance(message, MessageInterface)


# Does this need to be a Factor?
class MeanField(Collection, Dict[Variable, AbstractMessage], Factor):
    """For a factor with multiple variables, this class represents the
    the mean field approximation to that factor,

    f(x₁, x₂, x₃) = q(x₁, x₂, x₃) = q₁(x₁) q₂(x₂) q₃(x₃)

    Internally these variables approximations are stored in a
    dictionary with the variables as keys and the message or
    variable distribution as values


    Methods
    -------
    keys()
        returns the variables of the mean_field

    logpdf({x₁: x1, x₂: x2, x₃: x3})
        returns the q(x₁, x₂, x₃), axis defines the axes over which
        to reduce the return, if the meanfield is duplicated over multiple
        plates.

    logpdf_gradient({x₁: x1, x₂: x2, x₃: x3})
        returns the q(x₁, x₂, x₃) and the gradients for each input.
        to save memory the gradients are always the shape of the input
        values (i.e. this does not calculate the Jacobian)
    """

    def __init__(
        self,
        dists: Dict[Variable, Union[AbstractMessage, float]],
        plates: Optional[Tuple[Plate, ...]] = None,
        log_norm: np.ndarray = 0.0,
    ):
        dict.__init__(self, dists)
        Factor.__init__(self, self._logpdf, *self, arg_names=[])
        Collection.__init__(self)

        if isinstance(dists, MeanField):
            self.log_norm = dists.log_norm
            self._plates = dists.plates
        else:
            self.log_norm = log_norm
            self._plates = self.sorted_plates if plates is None else plates

    def copy(self) -> "MeanField":
        return type(self)(self)

    def __radd__(self, other):
        return self + other

    def __add__(self, other):
        return type(self)(
            {variable: message + other for variable, message in self.items()},
            plates=self.plates,
            log_norm=self.log_norm,
        )

    def __rsub__(self, other):
        return (-self) + other

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return type(self)(
            {variable: -message for variable, message in self.items()},
            plates=self.plates,
            log_norm=self.log_norm,
        )

    @cached_property
    def fixed_values(self):
        return VariableData(
            {k: dist.mean for k, dist in self.items() if isinstance(dist, FixedMessage)}
        )

    @classmethod
    def from_priors(cls, priors: Iterable[Prior]) -> "MeanField":
        """
        Create a MeanField from a list of priors.

        This works because priors are a kind of variable and
        messages can be derived from priors.

        Parameters
        ----------
        priors
            A list of priors

        Returns
        -------
        A mean field
        """
        return MeanField({prior: prior.message for prior in priors})

    pop = dict.pop
    values = dict.values
    items = dict.items
    __getitem__ = dict.__getitem__
    __len__ = dict.__len__

    @property
    def shapes(self):
        return {v: np.shape(m) for v, m in self.items()}

    @property
    def sizes(self):
        return {v: np.size(m) for v, m in self.items()}
    
    @property 
    def names(self):
        return {v.name: m for v, m in self.items()}

    def __getitem__(self, index):
        if isinstance(index, Variable):
            return dict.__getitem__(self, index)
        else:
            return self.subset(self.variables, plates_index=index)

    def __setitem__(self, index, value):
        if isinstance(index, Variable):
            dict.__setitem__(self, index, value)
        elif isinstance(value, MeanField):
            self.update_mean_field(value, index)

    def merge(self, index, mean_field):
        new_dist = dict(self)
        if index:
            plate_sizes = VariableData.plate_sizes(self)
            for v, message in mean_field.items():
                i = v.make_indexes(index, plate_sizes)
                new_dist[v] = new_dist[v].merge(i, message)
        else:
            new_dist.update(mean_field)

        return MeanField(new_dist)

    def update_mean_field(self, mean_field, plates_index=None):
        if plates_index:
            plate_sizes = VariableData.plate_sizes(self)
            for v, new_message in mean_field.items():
                index = v.make_indexes(plates_index, plate_sizes)
                self[v][index] = new_message
        else:
            self.update(mean_field)

        return self

    def subset(self, variables=None, plates_index=None):
        cls = type(self) if isinstance(self, MeanField) else MeanField
        variables = variables or self.variables
        if plates_index:
            plate_sizes = VariableData.plate_sizes(self)
            variable_index = (
                (v, v.make_indexes(plates_index, plate_sizes)) for v in variables
            )
            mean_field = dict((v, self[v][index]) for v, index in variable_index)

            return cls(mean_field)

        return cls((v, self[v]) for v in variables if v in self.keys())

    def rescale(self, rescale: Dict[Variable, float]) -> "MeanField":
        rescaled = {}
        for v, message in self.items():
            scale = rescale.get(v, 1)
            if scale == 1:
                rescaled[v] = message
            elif scale == 0:
                rescaled[v] = 1.0
            else:
                rescaled[v] = message ** scale

        return MeanField(rescaled)
    
    def map(self, func):
        return VariableData({v: func(dist) for v, dist in self.items()})

    def attr(self, attr: str):
        return MeanField.map(self, attrgetter(attr))

    @property
    def mean(self):
        return self.attr("mean")

    @property
    def variance(self):
        return self.attr("variance")

    @property
    def std(self):
        return self.attr("std")

    @property
    def scale(self):
        return self.attr("scale")

    @property
    def lower_limit(self):
        return self.map(
            lambda m: np.full(m.shape, m.lower_limit) if m.shape else m.lower_limit
        )

    @property
    def upper_limit(self):
        return self.map(
            lambda m: np.full(m.shape, m.upper_limit) if m.shape else m.upper_limit
        )
    
    def zeros_like(self):
        return MeanField.map(self, lambda m: m.zeros_like())

    def precision(self, variables=None):
        variables = variables or self.all_variables
        variances = MeanField.attr(self, "variance").subset(variables)
        return VariableFullOperator.from_diagonal(variances ** -1)

    @property
    def arguments(self) -> Dict[Variable, Prior]:
        """
        Arguments that can be used to update a Model
        """
        return {v: dist for v, dist in self.items()}

    def _logpdf(self, *args: np.ndarray) -> np.ndarray:
        return self.logpdf(dict(zip(self.args, args)))

    def logpdf(self, values: Dict[Variable, np.ndarray],) -> np.ndarray:
        """Calculates the logpdf of the passed values for messages

        the result is broadcast to the appropriate shape given the variable
        plates
        """
        return sum(np.sum(m.logpdf(values[v])) for v, m in self.items())

    def logpdf_gradient(self, values: Dict[Variable, np.ndarray], **kwargs):
        logl = 0
        gradl = {}
        for v, m in self.items():
            lv, gradl[v] = m.logpdf_gradient(values[v])
            logl += np.sum(lv)

        return logl, gradl

    def __repr__(self):
        reprdict = "{\n" + "\n".join(f"  {k}: {v}" for k, v in self.items()) + "\n  }"
        classname = type(self).__name__
        return f"{classname}({reprdict}, log_norm={self.log_norm})"

    @property
    def is_valid(self):
        return all(d.is_valid for d in self.values())

    def prod(self, *approxs: "MeanField", default=None) -> "MeanField":
        dists = [
            (
                key,
                prod(
                    (other_mean_field.get(key, 1.0) for other_mean_field in approxs),
                    message,
                ),
            )
            for key, message in self.items()
        ]
        if default is not None:
            dists = [
                (key, message if is_message(message) else default[key])
                for key, message in dists
            ]
        return MeanField({k: m for k, m in dists if is_message(m)})

    __mul__ = prod

    def __truediv__(self, other: "MeanField") -> "MeanField":
        # log_norm must be passed by keyword: the second positional ctor slot
        # is `plates`, and passing it positionally silently dropped the
        # evidence AND stored a float in `_plates` (#1332 F1).
        return type(self)(
            {k: m / other.get(k, 1.0) for k, m in self.items()},
            log_norm=self.log_norm - other.log_norm,
        )

    def __pow__(self, other: Union[float, "MeanField"]) -> "MeanField":
        if isinstance(other, Real):
            return type(self)(
                {k: m ** other for k, m in self.items()},
                log_norm=self.log_norm * other,
            )
        # Per-variable exponents: there is no meaningful scalar aggregate of
        # the two mean-field log_norms (the per-message log_norms carry
        # through each message's __pow__); the previous
        # `self.log_norm * other.log_norm` was not a meaningful quantity and
        # additionally landed in the plates slot (#1332 F1).
        return type(self)(
            {key: value ** other[key] for key, value in self.items()},
            log_norm=0.0,
        )

    def log_normalisation(self, other: "MeanField") -> float:
        return sum(np.sum(dist.log_normalisation(other[v])) for v, dist in self.items())

    def update_invalid(self, other: "MeanField") -> "MeanField":
        mean_field = {}
        for k, m in self.items():
            m2 = other.get(k)
            mean_field[k] = m.update_invalid(m2) if m2 else m

        return type(self)(mean_field, self.log_norm)

    def check_valid(self):
        return VariableData((v, m.check_valid()) for v, m in self.items())

    def project_mode(self, res: OptResult):
        return self.from_mode_covariance(res.mode, res.hess_inv, res.log_norm)

    def from_opt_state(self, state):
        return self.from_mode_covariance(
            state.all_parameters, state.full_hessian.inv(), state.value
        )

    def from_mode_covariance(
        self,
        mode: Dict[Variable, np.ndarray],
        covar: Dict[Variable, np.ndarray],
        fun: Optional[float] = None,
    ):
        """
        Projects the mode and covariance
        """
        mode = ChainMap(mode, self.fixed_values)
        projection = MeanField(
            {
                v: self[v].from_mode(
                    mode[v],
                    covar.get(v),
                    id_=self[v].id,
                    lower_limit=self[v].lower_limit,
                    upper_limit=self[v].upper_limit,
                )
                for v in self.keys() & mode.keys()
            }
        )
        if fun is not None:
            projection.log_norm = fun - projection(mode).log_value

        return projection

    def sample(self, n_samples=None):
        return VariableData({v: dist.sample(n_samples) for v, dist in self.items()})

    def kl(self, mean_field: "MeanField") -> np.ndarray:
        """
        The KL divergence between two mean fields, summed over variables:

            KL(self || mean_field) = Σᵢ KL( self[xᵢ] || mean_field[xᵢ] )

        which is exact because both mean fields factorise over variables.

        Direction contract: ``message.kl(other)`` means ``KL(message || other)``
        for every message family. ``EPHistory`` uses this sum between
        successive global mean fields as its convergence criterion.
        """
        return sum(np.sum(dist.kl(mean_field[k])) for k, dist in self.items())

    __hash__ = Factor.__hash__

    @classmethod
    def from_dist(
        cls, dist: Union[Dict[Variable, AbstractMessage], "MeanField"]
    ) -> "MeanField":
        return dist if isinstance(dist, cls) else MeanField(dist)

    def update_factor_mean_field(
        self,
        cavity_dist: "MeanField",
        last_dist: Optional["MeanField"] = None,
        delta: Delta = 1.0,
        status: Status = Status(),
    ) -> Tuple["MeanField", Status]:
        """
        The EP factor update: given ``self`` = the newly-fitted model
        distribution q* (the moment-matched projection of the tilted
        distribution), divide out the cavity to recover the factor's new
        message, damped by ``delta``:

            q_a_new = (q*)^δ · (q_a_old)^(1−δ) / (cavity)^δ

        which on natural parameters is an exponential moving average,

            η_a ← (1 − δ)·η_a + δ·(η_q* − η_cavity)

        ``delta`` may be a scalar or a per-variable ``MeanField`` of scalars
        (see ``DynamicUpdater``). At ``delta == 1`` this reduces to the
        undamped ``q* / cavity``.

        The natural-parameter subtraction is not closed in the family (it can
        produce e.g. a negative variance). If the result is invalid, the
        invalid parameters are reverted per-parameter to ``last_dist`` via
        ``update_invalid`` and the status is flagged ``BAD_PROJECTION``.

        Parameters
        ----------
        cavity_dist
            The cavity distribution: the product of every other factor's
            messages on this factor's variables.
        last_dist
            The factor's previous message distribution ``q_a_old`` (required
            when ``delta < 1`` and as the fallback for invalid projections).
        delta
            Damping coefficient in (0, 1]; scalar or per-variable MeanField.
        status
            Status carried through from the factor optimisation.

        Returns
        -------
        The factor's new message distribution and the updated status.
        """
        success, messages, _, flag = status
        updated = False
        try:
            with LogWarnings(
                logger=_log_projection_warnings, action="always"
            ) as caught_warnings:
                if isinstance(delta, MeanField) or delta < 1:
                    factor_dist = (
                        self ** delta * last_dist ** (1 - delta)
                    ) / cavity_dist ** delta
                else:
                    factor_dist = self / cavity_dist

            for m in caught_warnings.messages:
                messages += (f"project_mean_field warning: {m}",)

            if not factor_dist.is_valid:
                success = False
                messages += (f"model projection for {self} is invalid",)
                factor_dist = factor_dist.update_invalid(last_dist)
                # May want to check another way
                # e.g. factor_dist.check_valid().sum() / factor_dist.check_valid().size
                valid = factor_dist.check_valid()
                if valid.any():
                    updated = True
                    n_valid = valid.sum()
                    n_total = valid.size
                    logger.debug(
                        "meanfield with variables: %r ,"
                        "partially updated %d parameters "
                        "out of %d total, %.0%%",
                        tuple(self.variables),
                        n_valid,
                        n_total,
                        n_valid / n_total,
                    )

                flag = StatusFlag.BAD_PROJECTION
            else:
                updated = True

        except exc.MessageException as e:
            logger.exception(e)
            factor_dist = last_dist

        return (
            factor_dist,
            Status(
                success=success,
                messages=messages,
                updated=updated,
                flag=flag,
                result=status.result,
            ),
        )


class FactorApproximation(AbstractNode):
    """
    This class represents the 'tilted distribution' in EP,

    When approximating a model distribution of the form,

    m(x) = ∏ₐ fₐ(xₐ)

    we can define an approximating distribution as the product of
    factor distributions,

    q(x) = ∏ₐ qₐ(xₐ)

    the 'cavity distribution' q⁻ᵃ for a factor can be viewed as a
    prior distribution for the factor,

    q⁻ᵃ(xₐ) = ∏_{ᵦ ≠ a} qᵦ(xᵦ)

    so the model can be approximated by the 'tilted distribution'

    q⁺ᵃ(xₐ) = fₐ(xₐ) q⁻ᵃ(xₐ)

    Parameters
    ----------
    factor: Factor
        fₐ(xₐ)
    cavity_dist: MeanField
        q⁻ᵃ(xₐ)
    factor_dist: MeanField
        qₐ(xₐ)
    model_dist: MeanField
        q(xₐ)

    Methods
    -------
    __call__(values={xₐ: x₀})
        returns q⁺ᵃ(x₀)

    func_jacobian(values={xₐ: x₀}, variables=(xₐ,))
        returns q⁺ᵃ(x₀), {xₐ: dq⁺ᵃ(x₀)/dxₐ}

    project_mean_field(mean_field, delta=1., status=Status())
        for qᶠ = mean_field, finds qᶠₐ such that qᶠₐ * q⁻ᵃ = qᶠ
        delta controls how much to change from the previous factor
        distribution qₐ, so qʳₐ = (qᶠₐ)ᵟ * (qₐ)¹⁻ᵟ

        returns qʳₐ, status
    """

    def __init__(
        self,
        factor: Factor,
        cavity_dist: MeanField,
        factor_dist: MeanField,
        model_dist: MeanField,
    ):
        # Have to seperate FactorApproximation into two classes
        # in order to be able to redefine __new__
        self.factor = factor
        self.cavity_dist = MeanField.from_dist(cavity_dist)
        self.factor_dist = MeanField.from_dist(factor_dist)
        self.model_dist = MeanField.from_dist(model_dist)

        self.fixed_values = self.factor_dist.fixed_values

        super().__init__(**factor._kwargs)

    @property
    def variables(self):
        return self.factor.variables

    @property
    def deterministic_variables(self):
        return self.factor.deterministic_variables

    @property
    def all_variables(self):
        return self.factor.all_variables

    @property
    def mean_field(self):
        return self.model_dist

    @property
    def name(self):
        return f"FactorApproximation({self.factor.name})"

    @property
    def deterministic_dist(self):
        """
        the `MeanField` approximation of the deterministic variables
        """
        return MeanField({v: self.cavity_dist[v] for v in self.deterministic_variables})

    def __call__(self, values: Dict[Variable, np.ndarray],) -> FactorValue:
        variable_dict = {**self.fixed_values, **values}
        fval = self.factor(variable_dict)
        log_meanfield = self.cavity_dist({**variable_dict, **fval.deterministic_values})
        return np.sum(fval) + np.sum(log_meanfield)

    def func_jacobian(
        self, values: Dict[Variable, np.ndarray]
    ) -> Tuple[FactorValue, AbstractJacobian]:
        raise NotImplementedError()

    def func_gradient(
        self, values: Dict[Variable, np.ndarray],
    ) -> Tuple[FactorValue, VariableData]:
        variable_dict = {**self.fixed_values, **values}
        fval, fjac = self.factor.func_jacobian(variable_dict)

        variable_dict.update(fval.deterministic_values)
        log_cavity, grad_cavity = self.cavity_dist.logpdf_gradient(
            {**variable_dict, **fval.deterministic_values}
        )

        logl = np.sum(fval) + np.sum(log_cavity)
        grad = fjac.grad(grad_cavity)

        return logl, grad

    def project_mean_field(
        self, model_dist: MeanField, delta: Delta = 1.0, status: Status = Status(),
    ) -> Tuple["FactorApproximation", Status]:
        factor_dist, status = model_dist.update_factor_mean_field(
            self.cavity_dist, last_dist=self.factor_dist, delta=delta, status=status,
        )
        new_approx = FactorApproximation(
            self.factor,
            self.cavity_dist,
            factor_dist=factor_dist,
            model_dist=model_dist,
        )
        return new_approx, status

    project = project_mean_field

    def __repr__(self):
        # TODO make this nicer
        return f"{type(self).__name__}({self.factor}, ...)"
