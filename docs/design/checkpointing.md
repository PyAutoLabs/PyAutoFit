# Design note: archives, resume states and checkpointing

Status: search-extensibility phase A3 (decision D5). Implemented in
`autofit/non_linear/checkpoint.py`; the samples conversion that reads the archive is
`autofit/non_linear/samples/adapter.py`. This note fixes the meaning of the two kinds
of internal state, how they are written, what happens when they are corrupt, and which
files survive the end of a fit.

## Two kinds of internal state

A search's `search_internal/` folder used to hold one thing that meant two things: a
checkpoint an interrupted run resumes from, and the post-hoc archive of the backend's
results. The config key `output.search_internal: false` deletes the folder at the end
of a fit, so searches whose results could not be rebuilt without it (Emcee,
BlackJAX NUTS, SMC) forced the key on in their `__init__`, which leaked to every later
search in the process. A3 separates the two:

| Role | Class attribute | Default | Written | Read by |
|---|---|---|---|---|
| **archive** (`result_internal`) | `NonLinearSearch.checkpointer` | `DillCheckpointer()` | at the end of the fit (`output_search_internal` -> `checkpointer.finalize`), and by searches that also save it mid-run | `Result.search_internal`, `samples_via_internal_from` on a completed folder |
| **resume state** | `NonLinearSearch.resume_state` | `None` | by the backend during the run | the search's own resume path; `samples_via_internal_from` falls back to it when no archive exists yet |

Only searches that genuinely resume an interrupted run declare a resume state, and
`resumable = True` is declared only for them: Dynesty and Nautilus (native backend
checkpoints), NSS (its own pickle), MultiStart (whose archive dill is also its resume
state) and Emcee (whose HDF backend is both). BlackJAX NUTS and SMC start every `_fit`
afresh, so they are not resumable and have no resume state.

| Search | Archive | Resume state |
|---|---|---|
| Emcee | `NativeFileCheckpointer("search_internal.hdf", load_emcee_backend, retain_after_completion=True)` | the same file |
| Zeus, BFGS, LBFGS, Drawer | `DillCheckpointer()` | none |
| BlackJAX NUTS, SMC | `PickleCheckpointer()` (`search_internal.pickle`, retained) | none |
| DynestyStatic, DynestyDynamic | `DillCheckpointer()` | `NativeFileCheckpointer("savestate.save", <Sampler>.restore)` |
| Nautilus | `DillCheckpointer(strip_attributes=("pool_l", "pool_s"))` | `NativeFileCheckpointer("checkpoint.hdf5", ...)`; only nautilus can reopen it (inside a `Sampler` built with the run's prior and likelihood), so its loader raises `NotImplementedError` |
| NSS | `DillCheckpointer()` | `NativeFileCheckpointer("nss_checkpoint.pkl", _load_checkpoint)` (written, read and deleted by NSS's `run(ctx)`) |
| MultiStart* | `DillCheckpointer()` | the same file |

Every existing filename is kept, so output folders written before A3 still load.

## Atomic writes

Every file autofit writes itself is written atomically: `Checkpointer.save` dumps to a
sibling temporary file (`<name>.<pid>.tmp`) and `os.replace`s it into place
(`autofit.tools.util.open_atomic`). A write interrupted by an exception, a
`KeyboardInterrupt`, a full disk or a killed process therefore leaves the previous
file whole, and the temporary file is removed on the exception paths.

Native files are written by their backend, with the backend's own guarantees:
dynesty's `save_sampler` writes `savestate.save.tmp` and renames it; NSS's
`_save_checkpoint` uses the same tmp-and-rename; emcee's `HDFBackend` and nautilus's
`checkpoint.hdf5` are HDF5 files appended in place by `h5py`, so an interrupted write
can leave them unreadable. `NativeFileCheckpointer.save` and `finalize` are no-ops:
autofit never writes or rewrites a native file.

## Corruption

A corrupt file is never silently replaced by a different source of samples:

- `Checkpointer.load` raises whatever its deserialiser raises (`EOFError`,
  `pickle.UnpicklingError`, `OSError` from `h5py`, ...). It never converts a corrupt
  file into `FileNotFoundError`.
- `NonLinearSearch.samples_from` falls back to `samples.csv` (with a WARNING) only for
  `FileNotFoundError` and `NotImplementedError` (A0b's narrowed fallback), so a
  corrupt archive surfaces as an error rather than as silently older samples.
- On the resume path a search decides. MultiStart treats a corrupt dill as absent and
  starts a fresh run with a warning (the state is only a resume aid). Dynesty, Nautilus
  and NSS let their backend's error propagate, because resuming from a damaged
  checkpoint is not safe and starting over silently would discard the run; deleting
  the named file restarts the fit.

## Retention precedence

When a fit completes, `NonLinearSearch.post_fit_output` decides which files survive,
in this order:

1. **`checkpointer.retain_after_completion`.** When true (Emcee, BlackJAX NUTS, SMC),
   the archive is always kept: their results cannot be rebuilt without it. This
   replaces the three `conf.instance["output"]["search_internal"] = True` mutations
   those searches made in `__init__`; constructing a search never changes the
   configuration.
2. **`output.search_internal`** (`config/output.yaml`). Otherwise, `true` keeps the
   archive and `false` removes the whole `search_internal/` folder (archive, resume
   state and timer files).
3. **The resume state** is deleted once the fit completes, because there is nothing
   left to resume, unless it is the archive's own file (Emcee, MultiStart) or declares
   `retain_after_completion`.
4. **Zipping.** Whatever survives is archived in `<identifier>.zip` and restored by
   `paths.restore()`, so a completed folder unzipped later loads its archive again.

A retained archive is therefore always readable after completion: a completed Dynesty
folder keeps `search_internal.dill`, which is what `samples_via_internal_from` reads
(`savestate.save` is the resume state and is removed).

## Loading, and outputs written before A3

Loading is injected rather than hard-coded in `paths`: `DirectoryPaths` knows its
search (`paths.search`), and `DirectoryPaths.load_search_internal()` asks that
search's `checkpointer`. `Result.search_internal` and every `samples_via_internal_from`
go through it.

Format-aware legacy detection is kept, not replaced by a dill-only fallback. The emcee
probe that used to live inline in `DirectoryPaths.load_search_internal` ("a nasty
hack") moved to `checkpoint.load_legacy_search_internal(directory)`, which tries, in
order, `search_internal.hdf` (an emcee `HDFBackend`), `search_internal.pickle` (NUTS,
SMC) and `search_internal.dill`. It is used when the paths has no search (a folder
opened on its own) and when a checkpointer's own archive file is absent, so an output
folder written before its search chose a strategy still loads.

Per output kind:

| Paths | Archive save | Archive load |
|---|---|---|
| `DirectoryPaths` (and `SubDirectoryPaths`) | the strategy's file under `files/search_internal/` | the strategy, then the legacy probe |
| zipped output | restored from the `.zip` by `paths.restore()` | as `DirectoryPaths` |
| `DatabasePaths` | no-op (results live in the database) | `None` |
| `NullPaths` | no-op | `None` |
| summary-only output (no `search_internal/`, e.g. `output.search_internal: false`) | n/a | `FileNotFoundError`, so `samples_from` and `Result.samples` use `samples.csv` |

The aggregator never reads the archive: it rebuilds samples from `samples.csv` and
`samples_info.json["class_path"]`, which A3 leaves unchanged (the golden fixtures in
`test_autofit/non_linear/samples/golden/` pin `samples.csv` byte for byte).
