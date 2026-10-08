"""
Regenerate the golden ``samples.csv`` fixtures (search-extensibility phase A3).

Run ONLY on a checkout whose sample conversion is the reference: ``origin/main``
before the A3 samples adapter landed (generated at 0dbf258c4). It runs a small real
fit with each of Emcee, DynestyStatic, Nautilus, BlackJAXNUTS and SMC and stores
the backend's own internal state; then, in a fresh process, it converts that stored
state back through ``samples_via_internal_from`` (``cases.samples_from``) and writes
the ``samples.csv`` and ``samples_info.json`` the conversion produces.
``test_golden_samples.py`` replays the same conversion and requires ``samples.csv``
to match: same header and row order, values equal to floating-point roundoff.

    python test_autofit/non_linear/samples/golden/generate.py [case ...]

The internals are stored so that they load without the fit that produced them:

- ``emcee.hdf``: emcee's own ``HDFBackend`` file;
- ``dynesty.dill`` / ``nautilus.dill``: the sampler with its likelihood, prior
  transform and pools removed (conversion reads only the stored points, weights
  and bounds);
- ``nuts.pickle`` / ``smc.pickle``: the plain NumPy dicts those searches pickle.

``model.json`` is the fitted model, so the fixtures do not depend on the prior
configuration of the config directory the tests run under.
"""
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[3]


def fit(name):
    import numpy as np

    import autofit as af
    from autonerves import conf
    from autonerves.dictable import to_dict

    from test_autofit.non_linear.samples.golden.cases import CASES

    class_name, kwargs, filename, _ = CASES[name]

    rng = np.random.default_rng(1)
    xvalues = np.arange(50)
    data = 25.0 * np.exp(-0.5 * ((xvalues - 25.0) / 5.0) ** 2) + rng.normal(size=50)
    noise_map = np.ones(50)

    model = af.Model(af.ex.Gaussian)
    # Non-uniform priors so the log prior column, and the log-likelihood derived from
    # a log posterior (Emcee), are not trivially zero.
    model.centre = af.GaussianPrior(mean=25.0, sigma=10.0)
    model.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=50.0)
    model.sigma = af.LogUniformPrior(lower_limit=0.1, upper_limit=20.0)

    output = Path(tempfile.mkdtemp())
    conf.instance.push(
        new_path=str(Path(af.__file__).parent / "config"), output_path=str(output)
    )

    np.random.seed(1)
    use_jax = name in ("nuts", "smc")
    analysis = af.ex.Analysis(data=data, noise_map=noise_map, use_jax=use_jax)
    search = getattr(af, class_name)(name=f"golden_{name}", **kwargs)
    result = search.fit(model=model, analysis=analysis)

    directory = HERE / name
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir()

    if name == "emcee":
        import emcee

        search.paths.restore()
        internal = emcee.backends.HDFBackend(
            filename=str(search.paths.search_internal_path / "search_internal.hdf"),
            read_only=True,
        )
        state = emcee.backends.Backend()
        state.reset(internal.shape[0], internal.shape[1])
        state.chain = internal.get_chain()
        state.log_prob = internal.get_log_prob()
        state.accepted = np.asarray(internal.accepted)
        state.iteration = internal.iteration
        state.random_state = internal.random_state
        state.blobs = None
    else:
        internal = result.search_internal
        if name == "dynesty":
            results = internal.results
            state = {
                key: np.asarray(getattr(results, key))
                for key in ("samples", "logl", "logwt", "logz", "ncall")
            }
        elif name == "nautilus":
            parameters, log_weights, log_likelihoods = internal.posterior()
            state = {
                "parameters": parameters,
                "log_weights": log_weights,
                "log_likelihoods": log_likelihoods,
                "log_z": internal.log_z,
                "n_like": internal.n_like,
                "n_live": internal.n_live,
            }
        else:
            state = internal

    with open(directory / filename, "wb") as f:
        pickle.dump(state, f)

    # The stored form must convert exactly as the live internal does.
    from test_autofit.non_linear.samples.golden.cases import internal_from, search_from

    def table(search_internal):
        samples = search_from(name).samples_via_internal_from(
            model=model, search_internal=search_internal
        )
        path = output / "check.csv"
        samples.write_table(filename=path)
        return path.read_bytes()

    assert table(internal) == table(internal_from(name)), name

    with open(directory / "model.json", "w") as f:
        json.dump(to_dict(model), f, indent=4)

    shutil.rmtree(output, ignore_errors=True)


def convert(name):
    from autofit.non_linear.paths.null import NullPaths  # noqa: F401

    from test_autofit.non_linear.samples.golden.cases import (
        HERE as CASES_HERE,
        samples_from,
        samples_info_for_comparison,
    )

    samples = samples_from(name)
    samples.write_table(filename=CASES_HERE / name / "samples.csv")
    with open(CASES_HERE / name / "samples_info.json", "w") as f:
        json.dump(samples_info_for_comparison(samples), f, indent=4, sort_keys=True)


def main():
    from test_autofit.non_linear.samples.golden.cases import CASES

    names = sys.argv[2:] if len(sys.argv) > 2 else list(CASES)

    if len(sys.argv) > 1 and sys.argv[1] in ("fit", "convert"):
        action = sys.argv[1]
        for name in names:
            (fit if action == "fit" else convert)(name)
        return

    env = {**os.environ, "PYAUTO_SKIP_VISUALIZATION": "1"}
    env.pop("PYAUTO_TEST_MODE", None)
    names = sys.argv[1:] or list(CASES)
    for action in ("fit", "convert"):
        for name in names:
            # One fresh process per step: conversion must not see the fit's state.
            subprocess.run(
                [sys.executable, __file__, action, name], check=True, cwd=ROOT, env=env
            )


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
