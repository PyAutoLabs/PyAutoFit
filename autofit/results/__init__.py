"""
The results library (results-library epic): find, classify and count thousands of
finished fits without opening each one by hand.

- ``ResultsIndex``: an incremental index of every result under one or more roots, one
  JSON record per result (``ResultRecord``), each classified into a ``ResultState``;
- ``ResultsCache``: a local SQLite cache of an index for queries;
- ``ResultsLibrary``: what a set of fits is expected to hold, and ``check`` for what is
  missing (``CompletenessReport``);
- ``enrich``: add a product to finished results (``Aggregator.enrich``);
- ``python -m autofit.results {index,status,count,check,enrich}``: the command line.
"""
from autofit.non_linear.status.states import ResultState
from autofit.results.cache import ResultsCache
from autofit.results.classify import classify
from autofit.results.index import DEFAULT_LATEST_KEY, ResultsIndex, find_results
from autofit.results.record import ResultRecord
from autofit.results.completeness import CompletenessReport
from autofit.results.enrich import EnrichReport, enrich
from autofit.results.library import ExpectedResult, LibraryException, ResultsLibrary
