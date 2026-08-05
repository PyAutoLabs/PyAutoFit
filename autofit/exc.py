from autonerves.exc import PriorException


class MessageException(PriorException):
    """
    Raised when some assertion about the parameterization of a message is not met
    """


class PathsException(Exception):
    pass


class FitException(Exception):
    """
    An exception to be thrown if the non linear search must resample; equivalent to returning an infinitely bad fit
    """

    pass


class PipelineException(Exception):
    pass


class DeferredInstanceException(Exception):
    """
    Exception raised when an attempt is made to access an attribute or function of a
    deferred instance prior to instantiation
    """

    pass


class AggregatorException(Exception):
    pass


class GridSearchException(Exception):
    pass


class HistoryException(Exception):
    """
    Thrown when insufficient factor history is present for a given operation
    """


class InitializerException(Exception):
    """
    Raises exceptions associated with the `non_linear.initializer` module and `Initializer` classes.

    For example if all initial samples have identical figures of merit.
    """


class FactorOptimisationException(Exception):
    """
    Thrown when a single factor in an expectation propagation graph fails to
    optimise on too many consecutive sweeps.

    An individual failed factor update is not fatal — it is recorded as a
    failure and the sweep continues using that factor's previous message (see
    `graphical.expectation_propagation.optimiser.factor_step`). This is raised
    only when one factor has failed enough times in a row that continuing would
    mean converging on a stale message and reporting success.
    """


class SamplesException(Exception):
    pass


class SearchException(Exception):
    pass


class SamplesWarning(Warning):
    """
    Raises warnings associated with the `non_linear` module and `NonLinearSearch` classes.

    For example if the search is parallel but enviromental variables controlling multithreading are sub-optimal.
    """
    pass


class SearchWarning(Warning):
    pass