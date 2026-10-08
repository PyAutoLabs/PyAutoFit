"""
The golden samples cases: how each stored backend internal is loaded and converted.

Shared by ``generate.py`` (which writes the fixtures on the reference checkout) and
``test_golden_samples.py`` (which replays the conversion and compares the tables), so the
two can never drift apart.
"""
import json
import pickle
from pathlib import Path


HERE = Path(__file__).parent

#: name -> (search class path inside ``autofit``, constructor kwargs, internal file,
#: modules the conversion needs).
CASES = {
    "emcee": ("Emcee", dict(nwalkers=12, nsteps=400), "emcee.pickle", ("emcee",)),
    "dynesty": ("DynestyStatic", dict(nlive=40, maxcall=4000), "dynesty.pickle", ()),
    "nautilus": ("Nautilus", dict(n_live=100, n_like_max=4000), "nautilus.pickle", ()),
    "nuts": (
        "BlackJAXNUTS",
        dict(num_warmup=50, num_samples=60, num_chains=2),
        "nuts.pickle",
        ("jax", "blackjax"),
    ),
    "smc": ("SMC", dict(num_particles=200), "smc.pickle", ()),
}


def model_from(name):
    from autonerves.dictable import from_dict

    with open(HERE / name / "model.json") as f:
        return from_dict(json.load(f))


class DynestyResults:
    """
    The fields of a ``dynesty`` ``Results`` that sample conversion reads.
    """

    def __init__(self, samples, logl, logwt, logz, ncall):
        self.samples = samples
        self.logl = logl
        self.logwt = logwt
        self.logz = logz
        self.ncall = ncall


class DynestyInternal:
    """
    A stored ``dynesty`` sampler reduced to its ``results``: everything
    ``samples_via_internal_from`` reads, without the sampler's likelihood, prior
    transform, pool or bounds (which tie a pickled sampler to one dynesty version).
    """

    def __init__(self, results: dict):
        self.results = DynestyResults(**results)


class NautilusInternal:
    """
    A stored ``nautilus.Sampler`` reduced to what sample conversion reads: the
    output of ``posterior()`` and the ``log_z`` / ``n_like`` / ``n_live``
    attributes (the sampler's neural-network bounds make the real object ~15 MB).
    """

    def __init__(self, state: dict):
        self._posterior = (
            state["parameters"],
            state["log_weights"],
            state["log_likelihoods"],
        )
        self.log_z = state["log_z"]
        self.n_like = state["n_like"]
        self.n_live = state["n_live"]

    def posterior(self):
        return self._posterior


def internal_from(name):
    """
    The stored internal of case ``name`` in the form the search converts: emcee's own
    in-memory ``Backend`` (the same ``get_chain`` / ``get_log_prob`` slicing as its
    ``HDFBackend``), the reduced dynesty and nautilus samplers above, and the plain
    NumPy dicts NUTS and SMC pickle.
    """
    _, _, filename, _ = CASES[name]

    with open(HERE / name / filename, "rb") as f:
        state = pickle.load(f)

    if name == "dynesty":
        return DynestyInternal(state)
    if name == "nautilus":
        return NautilusInternal(state)
    return state


def search_from(name):
    import autofit as af

    class_name, kwargs, _, _ = CASES[name]
    return getattr(af, class_name)(**kwargs)


def samples_from(name):
    """
    Convert the stored internal of case ``name`` to a ``Samples`` object through the
    search's public ``samples_via_internal_from``, under ``NullPaths`` (no timer).
    """
    search = search_from(name)
    model = model_from(name)
    return search.samples_via_internal_from(
        model=model, search_internal=internal_from(name)
    )


def samples_info_for_comparison(samples):
    """
    ``samples_info`` without the run-dependent ``time`` key, JSON round-tripped so
    NumPy scalars compare as the plain values ``samples_info.json`` stores.
    """
    from autofit.tools.util import NumpyEncoder

    info = {k: v for k, v in samples.samples_info.items() if k != "time"}
    return json.loads(json.dumps(info, cls=NumpyEncoder))
