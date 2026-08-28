from autonerves import jax_wrapper
from autonerves.dictable import register_parser
from . import conf

conf.instance.register(__file__)

import abc
import pickle
from dill import register

from . import exc
from . import mock as m
from .non_linear.grid.grid_search import GridSearch as SearchGridSearch
from .aggregator.base import AggBase
from .graphical.expectation_propagation.history import EPHistory
from .graphical.declarative.factor.analysis import AnalysisFactor
from .graphical.declarative.factor.analysis import EPAnalysisFactor
from .graphical.declarative.collection import FactorGraphModel
from .graphical.declarative.factor.hierarchical import HierarchicalFactor
from .graphical.expectation_propagation.optimiser import (
    ApproxUpdater,
    DynamicUpdater,
    FactorUpdater,
    SimplerUpdater,
)
from .graphical.laplace import LaplaceOptimiser
from .non_linear.grid.grid_list import GridList
from .non_linear.samples.summary import SamplesSummary
from .non_linear.samples import SamplesMCMC
from .non_linear.samples import SamplesNest
from .non_linear.samples import Samples
from .non_linear.samples import SamplesPDF
from .non_linear.samples import Sample
from .non_linear.samples import load_from_table
from .non_linear.samples import SamplesStored
from .aggregator.summary.aggregate_csv import AggregateCSV
from .aggregator.summary.aggregate_csv import ValueType
from .aggregator.summary.aggregate_images import AggregateImages
from .aggregator.summary.aggregate_fits import AggregateFITS
from autofit.aggregator.fit_interface import Fit
from .aggregator.search_output import SearchOutput
from .mapper import prior
from .mapper.model import AbstractModel
from .mapper.model import ModelInstance
from .mapper.model import ModelInstance as Instance
from .mapper.model import path_instances_of_class
from .mapper.model_mapper import ModelMapper
from .mapper.model_mapper import ModelMapper as Mapper
from .mapper.model_object import ModelObject
from .mapper.operator import DiagonalMatrix
from .mapper.prior.constant import Constant
from .mapper.prior.arithmetic.assertion import ComparisonAssertion
from .mapper.prior.arithmetic.assertion import ComparisonAssertion
from .mapper.prior.arithmetic.assertion import GreaterThanLessThanAssertion
from .mapper.prior.arithmetic.assertion import GreaterThanLessThanEqualAssertion
from .mapper.prior.deferred import DeferredArgument
from .mapper.prior.deferred import DeferredInstance
from .mapper.prior.width_modifier import AbsoluteWidthModifier
from .mapper.prior.width_modifier import RelativeWidthModifier
from .mapper.prior.width_modifier import WidthModifier
from .mapper.prior import GaussianPrior
from .mapper.prior import LogGaussianPrior
from .mapper.prior import LogUniformPrior
from .mapper.prior import TruncatedGaussianPrior
from .mapper.prior.vectorized import PriorVectorized
from .mapper.prior.abstract import Prior
from .mapper.prior.tuple_prior import TuplePrior
from .mapper.prior import UniformPrior
from .mapper.prior_model.abstract import AbstractPriorModel
from .mapper.prior_model.annotation import AnnotationPriorModel
from .mapper.prior_model.collection import Collection
from .mapper.prior_model.prior_model import Model
from .mapper.prior_model.array import Array
from .non_linear.search.abstract_search import NonLinearSearch
from .non_linear.analysis.visualize import Visualizer
from .non_linear.analysis.latent import Latent
from .non_linear.analysis.analysis import Analysis
from .non_linear.grid.grid_search import GridSearchResult
from .non_linear.grid.sensitivity import Sensitivity
from .non_linear.clipper import AbstractClipper
from .non_linear.clipper import ClipperNone
from .non_linear.clipper import ClipperPriorBox
from .non_linear.clipper import ClipperPriorBoxJoint
from .non_linear.scaler import AbstractScaler
from .non_linear.scaler import ScalerNone
from .non_linear.scaler import ScalerPriorWidth
from .non_linear.bijector import AbstractBijector
from .non_linear.bijector import BijectorNone
from .non_linear.bijector import BijectorAuto
from .non_linear.bijector import BijectorLogit
from .non_linear.bijector import BijectorPerPath
from .non_linear.bijector import BijectorDiagonal
from .non_linear.initializer import InitializerBall
from .non_linear.initializer import InitializerPrior
from .non_linear.initializer import InitializerParamBounds
from .non_linear.initializer import InitializerParamStartPoints
from .non_linear.search.mcmc.auto_correlations import AutoCorrelationsSettings
from .non_linear.search.mcmc.blackjax.nuts.search import BlackJAXNUTS
from .non_linear.search.mcmc.emcee.search import Emcee
from .non_linear.search.mcmc.zeus.search import Zeus
from .non_linear.search.nest.nautilus.search import Nautilus
from .non_linear.search.nest.dynesty.search.dynamic import DynestyDynamic
from .non_linear.search.nest.dynesty.search.static import DynestyStatic
from .non_linear.search.mle.drawer.search import Drawer
from .non_linear.search.mle.bfgs.search import BFGS
from .non_linear.search.mle.bfgs.search import LBFGS
from .non_linear.search.mle.multi_start_gradient.search import MultiStartAdam
from .non_linear.search.mle.multi_start_gradient.search import MultiStartADABelief
from .non_linear.search.mle.multi_start_gradient.search import MultiStartLion
from .non_linear.search.mle.multi_start_gradient.search import MultiStartProdigy
from .non_linear.search.mle.multi_start_gradient.convergence import (
    MultiStartGradientConvergence,
)
from .non_linear.paths.abstract import AbstractPaths
from .non_linear.paths import DirectoryPaths
from .non_linear.paths import DatabasePaths
from .non_linear.result import Result
from .non_linear.result import ResultsCollection
from .non_linear.settings import SettingsSearch
from .non_linear.samples.pdf import marginalize
from .text import formatter
from .text import samples_text
from .visualise import VisualiseGraph
from .interpolator import (
    LinearInterpolator,
    SplineInterpolator,
    CovarianceInterpolator,
    LinearRelationship,
)
from .tools import util

from autofit.mapper.prior.arithmetic.compound import SumPrior as Add
from autofit.mapper.prior.arithmetic.compound import MultiplePrior as Multiply
from autofit.mapper.prior.arithmetic.compound import DivisionPrior as Divide
from autofit.mapper.prior.arithmetic.compound import ModPrior as Mod
from autofit.mapper.prior.arithmetic.compound import PowerPrior as Power
from autofit.mapper.prior.arithmetic.compound import AbsolutePrior as Abs
from autofit.mapper.prior.arithmetic.compound import Log
from autofit.mapper.prior.arithmetic.compound import Log10

from . import example as ex


for type_ in (
    "model",
    "collection",
    "tuple_prior",
    "dict",
    "instance",
    "Uniform",
    "LogUniform",
    "Gaussian",
    "LogGaussian",
    "TruncatedGaussian",
    "compound",
    "Constant",
):
    register_parser(type_, ModelObject.from_dict)


@register(abc.ABCMeta)
def save_abc(pickler, obj):
    pickle._Pickler.save_type(pickler, obj)


# Last manual sync of the source stamp. Release wheels are stamped at build
# time and the git tag is the release truth — deliberately never bumped per
# release (PyAutoBuild#118/#120).
__version__ = "2026.8.17.1"

from autonerves import check_version

check_version(__version__)

# ---------------------------------------------------------------------------
# Public re-export of the autonerves configuration / serialization surface.
#
# Workspaces, tutorials and downstream code import these names from the science
# library (e.g. ``from autolens import conf``) rather than depending on the
# ``autonerves`` package directly, so the underlying configuration / serialization
# layer stays an implementation detail of the library.
# ---------------------------------------------------------------------------
# ``conf`` is already exported above (``from . import conf``); the names below
# complete the surface.
from autonerves import jax_wrapper
from autonerves import fitsable
from autonerves import setup_colab
from autonerves import setup_notebook
from autonerves.conf import with_config
from autonerves.dictable import from_dict, from_json, to_dict, output_to_json
from autonerves.fitsable import (
    output_to_fits,
    hdu_list_for_output_from,
    ndarray_via_fits_from,
    ndarray_via_hdu_from,
    header_obj_from,
)
from autonerves.test_mode import (
    with_test_mode_segment,
    skip_visualization,
    skip_fit_output,
    skip_checks,
    is_test_mode,
    test_mode_level,
)

# Lazy attributes (PEP 562): NSS pulls blackjax -> jax, and the database
# aggregator pulls sqlalchemy + the declarative models — together over a
# second of import time that most sessions never use.
_LAZY_ATTRS = {
    "NSS": ("autofit.non_linear.search.nest.nss.search", "NSS"),
    "Aggregator": ("autofit.database.aggregator", "Aggregator"),
    "Query": ("autofit.database.aggregator", "Query"),
    "GridSearchAggregator": (
        "autofit.database.aggregator.aggregator",
        "GridSearchAggregator",
    ),
    "db": ("autofit.database", None),
}


def __getattr__(name):
    try:
        module_name, attr = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_name)
    value = module if attr is None else getattr(module, attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))
