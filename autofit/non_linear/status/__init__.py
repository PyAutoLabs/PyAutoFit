"""
The per-fit status record (hpc-campaign phase 1).

Every fit with an output folder writes ``<output_path>.status.json`` beside that folder
(so it survives ``output.remove_files``): a compact (at most 2 KB), atomically replaced
document with the fit's state, attempt, progress, resources and scheduler job, written
at start, every ``general.hpc.status_interval_s`` seconds, at each full update, on
SIGUSR1 / SIGTERM, on completion and on failure. ``<output_path>.status.jsonl`` keeps
the attempt history and ``<output_path>.traceback.txt`` the last failure's traceback.
``general.hpc.status: false`` turns all of it off.

The ``progress`` protocol: a search may define ``progress(search_internal)``, returning
a dict of whichever of ``n_like``, ``n_like_max``, ``n_live``, ``log_z``,
``max_log_likelihood``, ``f_live``, ``n_eff`` and ``fraction`` (0-1) it knows, or
``None``. It is called at each full update (``FitContext.update``); the record adds
``eta_s`` from the fraction's rate.

``states`` is the single vocabulary of run states, result states, failure classes and
product names. ``identity`` writes ``dataset_identity.json`` (results-library phase 1).
"""
from autofit.non_linear.status.states import (
    COLLECTOR_STATES,
    TERMINAL_STATES,
    WRITTEN_STATES,
    FailureClass,
    Product,
    ResultState,
    RunState,
    failure_class_of,
)
from autofit.non_linear.status.writer import MAX_BYTES, SCHEMA, StatusWriter
from autofit.non_linear.status.record import (
    FitStatus,
    status_enabled,
    status_notify,
    status_recorded,
    status_interval_s,
)
from autofit.non_linear.status.identity import (
    dataset_identity_enabled,
    write_dataset_identity,
)
