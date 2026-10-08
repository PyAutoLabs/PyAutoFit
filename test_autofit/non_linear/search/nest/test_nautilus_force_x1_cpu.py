"""
``Nautilus(force_x1_cpu=True)`` with a numpy analysis runs on one CPU without the JAX
vectorised likelihood (search-extensibility phase A1).

``force_x1_cpu`` routes the fit through ``fit_x1_cpu``, whose ``Fitness`` was built
with ``use_jax_vmap=self.use_jax_vmap`` (default ``True``) whatever the analysis, so a
numpy analysis was batched through ``jax.vmap`` and crashed. The vectorised path is
now requested only when the analysis is JAX.
"""

import numpy as np
import pytest

import autofit as af

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


class NumpyAnalysis(af.Analysis):
    def log_likelihood_function(self, instance):
        vector = np.array([instance.centre, instance.normalization, instance.sigma])
        return float(-0.5 * np.sum((vector - np.array([50.0, 25.0, 10.0])) ** 2))


def test_force_x1_cpu_runs_with_a_numpy_analysis(monkeypatch):
    monkeypatch.setenv("PYAUTO_TEST_MODE", "1")
    search = af.Nautilus(force_x1_cpu=True)

    assert search.use_jax_vmap is True

    result = search.fit(model=af.Model(af.ex.Gaussian), analysis=NumpyAnalysis())

    assert isinstance(result, af.Result)
    assert len(result.samples.sample_list) > 0
