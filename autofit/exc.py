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


class SamplesException(Exception):
    pass


class SearchException(Exception):
    pass


class ProjectionException(ValueError):
    """
    Raised when an importance-weighted message projection
    (`AbstractMessage.project`) produces non-finite sufficient statistics —
    from non-finite samples, nan / +inf log weights, all-zero weights, or
    overflow of the weighted moments.

    A `ValueError` subclass, deliberately: EP's `factor_step` recovers from
    it by keeping the factor's previous message for that sweep (a failed
    sweep update, not a failed graph fit). It is not a `MessageException`,
    which several callers turn into a silent revert or a -inf likelihood, and
    not a `SearchException`, which signals a misconfigured search.
    """


class SamplesWarning(Warning):
    """
    Raises warnings associated with the `non_linear` module and `NonLinearSearch` classes.

    For example if the search is parallel but enviromental variables controlling multithreading are sub-optimal.
    """
    pass


class SearchWarning(Warning):
    pass