from autoconf import jax_wrapper
from autoconf.dictable import register_parser
from . import conf

conf.instance.register(__file__)

import abc
import pickle
from dill import register

from . import exc
from . import mock as m
from .non_linear.grid.grid_search import GridSearch as SearchGridSearch
from .aggregator.base import AggBase
from .database.aggregator.aggregator import GridSearchAggregator
from .graphical.expectation_propagation.history import EPHistory
from .graphical.declarative.factor.analysis import AnalysisFactor
from .graphical.declarative.collection import FactorGraphModel
from .graphical.declarative.factor.hierarchical import HierarchicalFactor
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
from .database.aggregator import Aggregator
from .aggregator.summary.aggregate_csv import AggregateCSV
from .aggregator.summary.aggregate_csv import ValueType
from .aggregator.summary.aggregate_images import AggregateImages
from .aggregator.summary.aggregate_fits import AggregateFITS
from .database.aggregator import Query
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
from .non_linear.analysis.analysis import Analysis
from .non_linear.grid.grid_search import GridSearchResult
from .non_linear.grid.sensitivity import Sensitivity
from .non_linear.initializer import InitializerBall
from .non_linear.initializer import InitializerPrior
from .non_linear.initializer import InitializerParamBounds
from .non_linear.initializer import InitializerParamStartPoints
from .non_linear.search.mcmc.auto_correlations import AutoCorrelationsSettings
from .non_linear.search.mcmc.emcee.search import Emcee
from .non_linear.search.mcmc.zeus.search import Zeus
from .non_linear.search.nest.nautilus.search import Nautilus
from .non_linear.search.nest.dynesty.search.dynamic import DynestyDynamic
from .non_linear.search.nest.dynesty.search.static import DynestyStatic
from .non_linear.search.nest.ultranest.search import UltraNest
from .non_linear.search.mle.drawer.search import Drawer
from .non_linear.search.mle.bfgs.search import BFGS
from .non_linear.search.mle.bfgs.search import LBFGS
from .non_linear.search.mle.pyswarms.search.globe import PySwarmsGlobal
from .non_linear.search.mle.pyswarms.search.local import PySwarmsLocal
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
from . import database as db


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


__version__ = "2025.12.21.1"