"""EP evidence reports non-finite terms by name (PyAutoFit#1713).

The degenerate fixture injects the non-finite log-partition that triggered the
reporting bug. The production tail message now has a finite normalizer.
"""

import logging
import warnings

import numpy as np
import pytest

from autofit import graphical as g
from autofit.graphical.expectation_propagation import ep_mean_field
from autofit.graphical.expectation_propagation.diagnostics import (
    EPDiagnostics,
    check_evidence_nonfinite,
)
from autofit.graphical.expectation_propagation.history import FactorHistory
from autofit.mapper.variable import Variable
from autofit.messages.truncated_normal import TruncatedNormalMessage

LOGGER = "autofit.graphical.expectation_propagation.ep_mean_field"


def _ok(x):
    return x


def _degenerate(x):
    return -x


class NonfinitePartitionMessage(TruncatedNormalMessage):
    def log_partition(self, xp=np):
        return -xp.inf


def make_approx(x, mean, sigma, message_type=TruncatedNormalMessage):
    ok_factor = g.Factor(_ok, x, name="f_ok")
    degenerate_factor = g.Factor(_degenerate, x, name="f_degenerate")
    ok = TruncatedNormalMessage(mean=10.0, sigma=5.0, lower_limit=0.0, upper_limit=100.0)
    other = message_type(mean=mean, sigma=sigma, lower_limit=0.0, upper_limit=100.0)
    return g.EPMeanField(
        factor_graph=g.FactorGraph([ok_factor, degenerate_factor]),
        factor_mean_field={
            ok_factor: g.MeanField({x: ok}),
            degenerate_factor: g.MeanField({x: other}),
        },
    )


@pytest.fixture(name="x")
def make_x():
    return Variable("x")


@pytest.fixture(name="degenerate")
def make_degenerate(x):
    return make_approx(x, -7374.18, 384.36, NonfinitePartitionMessage)


@pytest.fixture(name="healthy")
def make_healthy(x):
    return make_approx(x, 20.0, 8.0)


def test_tail_message_has_finite_evidence(x):
    approximation = make_approx(x, -7374.18, 384.36)
    assert np.isfinite(approximation.log_evidence)
    assert approximation.nonfinite_evidence_terms == {}


def test_log_evidence_nan_without_runtime_warning(degenerate):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        log_evidence = degenerate.log_evidence

    assert isinstance(log_evidence, float)
    assert np.isnan(log_evidence)


def test_one_named_warning_per_run(degenerate, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        for _ in range(3):
            degenerate.log_evidence
        # A fresh EPMeanField over the same graph (what each EP update makes)
        # does not repeat the warning.
        g.EPMeanField(
            factor_graph=degenerate.factor_graph,
            factor_mean_field=degenerate.factor_mean_field,
        ).log_evidence

    records = [r for r in caplog.records if r.name == LOGGER]
    assert len(records) == 1
    assert "'x'" in records[0].getMessage()
    assert "+inf" in records[0].getMessage()


def test_nonfinite_evidence_terms_names_the_variable(degenerate):
    assert degenerate.nonfinite_evidence_terms == {"variable 'x'": "+inf"}


def test_healthy_evidence_unchanged(healthy, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            log_evidence = healthy.log_evidence

    variable_evidence = {
        v: np.sum(logz) for v, logz in healthy.variable_evidence.items()
    }
    expected = sum(
        np.sum(meanfield.log_norm)
        - sum(variable_evidence[v] for v in factor.all_variables)
        for factor, meanfield in healthy.factor_mean_field.items()
    ) + sum(variable_evidence.values())

    assert np.isfinite(log_evidence)
    assert log_evidence == expected
    assert healthy.nonfinite_evidence_terms == {}
    assert not [r for r in caplog.records if r.name == LOGGER]


def test_evidence_divergence_inf_on_nan(degenerate, healthy):
    factor = next(iter(healthy.factor_mean_field))
    history = FactorHistory(factor)
    history(healthy, g.Status(success=True))
    history(degenerate, g.Status(success=True))

    assert history.evidence_divergence() == float("inf")


def test_evidence_divergence_finite_when_healthy(healthy):
    factor = next(iter(healthy.factor_mean_field))
    history = FactorHistory(factor)
    history(healthy, g.Status(success=True))
    history(healthy, g.Status(success=True))

    assert history.evidence_divergence() == 0.0


def test_diagnostics_record_evidence_nonfinite(degenerate, healthy):
    diagnostics = EPDiagnostics()
    factor = next(iter(healthy.factor_mean_field))
    diagnostics.snapshot(factor, healthy, g.Status(success=True))
    assert check_evidence_nonfinite(diagnostics) == []

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        diagnostics.snapshot(factor, degenerate, g.Status(success=True))
        diagnostics.snapshot(factor, degenerate, g.Status(success=True))

    assert np.isnan(diagnostics.factor_rows[-1]["log_evidence"])
    lines = check_evidence_nonfinite(diagnostics)
    assert len(lines) == 1
    assert lines[0].startswith("evidence_nonfinite: ")
    assert "variable 'x' (+inf, first at step 1)" in lines[0]


@pytest.fixture(autouse=True)
def reset_reported():
    ep_mean_field._reported_nonfinite.clear()
    yield
    ep_mean_field._reported_nonfinite.clear()
