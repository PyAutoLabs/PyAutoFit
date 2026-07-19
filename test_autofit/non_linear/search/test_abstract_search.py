import os

import numpy as np
import pytest

import autofit as af
from autonerves import conf

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(name="mapper")
def make_mapper():
    return af.ModelMapper()


@pytest.fixture(name="mock_list")
def make_mock_list():
    return [af.Model(af.m.MockClassx4), af.Model(af.m.MockClassx4)]


@pytest.fixture(name="result")
def make_result():
    mapper = af.ModelMapper()
    mapper.component = af.m.MockClassx2Tuple
    # noinspection PyTypeChecker
    return af.mock.MockResult(
        samples_summary=af.m.MockSamplesSummary(
            model=mapper,
            prior_means=[0, 1],
        ),
    )


def test__environment_variable_override():
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
    os.environ["NUMEXPR_NUM_THREADS"] = "2"

    conf.instance["general"]["parallel"]["warn_environment_variables"] = True

    with pytest.warns(af.exc.SearchWarning):
        af.mock.MockSearch(number_of_cores=2)

    conf.instance["general"]["parallel"]["warn_environment_variables"] = False

class TestResult:
    def test_model(self, result):

        component = result.model.component
        assert component.one_tuple.one_tuple_0.mean == 0.5
        assert component.one_tuple.one_tuple_1.mean == 1

    def test_model_centred(self, result):

        component = result.model_centred.component
        assert component.one_tuple.one_tuple_0.mean == 0
        assert component.one_tuple.one_tuple_1.mean == 1
        assert component.one_tuple.one_tuple_0.sigma == 0.2
        assert component.one_tuple.one_tuple_1.sigma == 0.2

    def test_model_centred_absolute(self, result):
        component = result.model_centred_absolute(a=2.0).component
        assert component.one_tuple.one_tuple_0.mean == 0
        assert component.one_tuple.one_tuple_1.mean == 1
        assert component.one_tuple.one_tuple_0.sigma == 2.0
        assert component.one_tuple.one_tuple_1.sigma == 2.0

    def test_model_centred_relative(self, result):
        component = result.model_centred_relative(r=1.0).component
        assert component.one_tuple.one_tuple_0.mean == 0
        assert component.one_tuple.one_tuple_1.mean == 1
        assert component.one_tuple.one_tuple_0.sigma == 0.0
        assert component.one_tuple.one_tuple_1.sigma == 1.0

    def test_raises(self, result):
        with pytest.raises(af.exc.PriorException):
            result.model.mapper_from_prior_means(
                result.samples_summary.prior_means, a=2.0, r=1.0
            )


class TestSearchConfig:
    def test__explicit_params_accessible(self):
        search = af.DynestyStatic(nlive=100)
        assert search.nlive == 100

    def test__run_params_accessible(self):
        search = af.DynestyStatic(dlogz=0.5)
        assert search.dlogz == 0.5

    def test__unique_tag(self):
        search = af.DynestyStatic(unique_tag="my_tag")
        assert search.unique_tag == "my_tag"

    def test__path_prefix_and_name(self):
        from pathlib import Path

        search = af.DynestyStatic(
            path_prefix="prefix",
            name="my_search",
        )
        assert search.path_prefix == Path("prefix")
        assert search.name == "my_search"

    def test__identifier_fields_differ_across_searches(self):
        emcee = af.Emcee()
        dynesty = af.DynestyStatic()

        assert emcee.__identifier_fields__ != dynesty.__identifier_fields__
        assert "nwalkers" in emcee.__identifier_fields__
        assert "nlive" in dynesty.__identifier_fields__

    def test__bypass_fake_samples_support_multi_batch_checks(self):
        model = af.Model(af.m.MockClassx2)
        parameter_vector = [1.0, 2.0]

        sample_list = af.DynestyStatic._build_fake_samples(
            model=model,
            parameter_vector=parameter_vector,
            log_likelihood=-10.0,
        )

        assert len(sample_list) == 4
        assert sample_list[0].parameter_lists_for_model(model) == parameter_vector
        assert sample_list[0].log_likelihood == -10.0
        assert [sample.log_likelihood for sample in sample_list] == [
            -10.0,
            -11.0,
            -12.0,
            -13.0,
        ]
        assert all(sample.weight > 0.0 for sample in sample_list)


class TestBypassFakeSamplesSizeRealistic:
    """
    PYAUTO_TEST_MODE_SAMPLES=N makes the bypass write N samples so
    samples.csv row count / byte size match a production sampler stage
    (PyAutoFit#1379; design locked on #1378).
    """

    def _samples_pdf(self, model, sample_list):
        from autofit.non_linear.samples.pdf import SamplesPDF

        return SamplesPDF(
            model=model,
            sample_list=sample_list,
            samples_info={
                "total_iterations": 1,
                "time": 0.0,
                "log_evidence": -10.0,
            },
        )

    def test__env_unset__legacy_four_samples_byte_identical(self, monkeypatch):
        monkeypatch.delenv("PYAUTO_TEST_MODE_SAMPLES", raising=False)

        model = af.Model(af.m.MockClassx2)
        parameter_vector = [1.0, 2.0]

        sample_list = af.DynestyStatic._build_fake_samples(
            model=model,
            parameter_vector=parameter_vector,
            log_likelihood=-10.0,
        )

        assert len(sample_list) == 4
        assert sample_list[0].parameter_lists_for_model(model) == [1.0, 2.0]
        assert sample_list[1].parameter_lists_for_model(model) == [1.001, 2.002]
        assert sample_list[2].parameter_lists_for_model(model) == [0.999, 1.998]
        assert sample_list[3].parameter_lists_for_model(model) == [1.002, 2.004]
        assert [sample.weight for sample in sample_list] == [1.0, 0.5, 0.25, 0.125]

    def test__env_below_four__raises(self, monkeypatch):
        monkeypatch.setenv("PYAUTO_TEST_MODE_SAMPLES", "3")

        with pytest.raises(ValueError):
            af.DynestyStatic._build_fake_samples(
                model=af.Model(af.m.MockClassx2),
                parameter_vector=[1.0, 2.0],
                log_likelihood=-10.0,
            )

    def test__large_n__structure_and_determinism(self, monkeypatch):
        monkeypatch.setenv("PYAUTO_TEST_MODE_SAMPLES", "50")

        model = af.Model(af.m.MockClassx2)
        parameter_vector = [1.0, 2.0]

        sample_list = af.DynestyStatic._build_fake_samples(
            model=model,
            parameter_vector=parameter_vector,
            log_likelihood=-10.0,
        )

        assert len(sample_list) == 50

        # Best sample is the unperturbed prior median, first, as in the
        # 4-sample branch; likelihoods are monotone decreasing from it.
        assert sample_list[0].parameter_lists_for_model(model) == [1.0, 2.0]
        assert sample_list[0].log_likelihood == -10.0
        assert sample_list[-1].log_likelihood == -10.0 - 49.0

        weights = [sample.weight for sample in sample_list]
        assert all(w > 0.0 for w in weights)
        assert weights == sorted(weights, reverse=True)
        assert sum(weights) == pytest.approx(1.0)

        # Perturbed parameters stay within the 1e-3 scatter scale.
        for sample in sample_list[1:]:
            params = sample.parameter_lists_for_model(model)
            assert params[0] == pytest.approx(1.0, abs=1e-2)
            assert params[1] == pytest.approx(2.0, abs=2e-2)

        # Fixed seed: a second call reproduces the set exactly.
        sample_list_repeat = af.DynestyStatic._build_fake_samples(
            model=model,
            parameter_vector=parameter_vector,
            log_likelihood=-10.0,
        )
        assert [
            s.parameter_lists_for_model(model) for s in sample_list_repeat
        ] == [s.parameter_lists_for_model(model) for s in sample_list]

    def test__zero_valued_parameter__perturbed_like_legacy_branch(
        self, monkeypatch
    ):
        monkeypatch.setenv("PYAUTO_TEST_MODE_SAMPLES", "20")

        model = af.Model(af.m.MockClassx2)

        sample_list = af.DynestyStatic._build_fake_samples(
            model=model,
            parameter_vector=[0.0, 2.0],
            log_likelihood=-10.0,
        )

        for sample in sample_list[1:]:
            params = sample.parameter_lists_for_model(model)
            assert params[0] != 0.0
            assert abs(params[0]) < 1e-2

    def test__summary_and_median_pdf_on_synthetic_set(self, monkeypatch):
        monkeypatch.setenv("PYAUTO_TEST_MODE_SAMPLES", "50")

        model = af.Model(af.m.MockClassx2)

        sample_list = af.DynestyStatic._build_fake_samples(
            model=model,
            parameter_vector=[1.0, 2.0],
            log_likelihood=-10.0,
        )
        samples = self._samples_pdf(model=model, sample_list=sample_list)

        # Max weight ~10/N < 0.99, so the real weighted-quantile path runs
        # (the 4-sample branch's max weight 1.0 forces the unconverged
        # fallback) — this is the production-representative code path.
        assert samples.pdf_converged is True

        median_pdf = samples.median_pdf(as_instance=False)
        assert median_pdf[0] == pytest.approx(1.0, abs=1e-2)
        assert median_pdf[1] == pytest.approx(2.0, abs=2e-2)

        summary = samples.summary()
        assert summary.max_log_likelihood_sample.log_likelihood == -10.0

        lower, upper = samples.values_at_sigma(sigma=1.0, as_instance=False)
        assert all(np.isfinite(lower)) and all(np.isfinite(upper))
        lower, upper = samples.values_at_sigma(sigma=3.0, as_instance=False)
        assert all(np.isfinite(lower)) and all(np.isfinite(upper))

        instance = samples.max_log_likelihood()
        assert instance.one == 1.0
        assert instance.two == 2.0

    def test__write_table_round_trip(self, monkeypatch, tmp_path):
        import csv

        monkeypatch.setenv("PYAUTO_TEST_MODE_SAMPLES", "100")

        model = af.Model(af.m.MockClassx2)

        sample_list = af.DynestyStatic._build_fake_samples(
            model=model,
            parameter_vector=[1.0, 2.0],
            log_likelihood=-10.0,
        )
        samples = self._samples_pdf(model=model, sample_list=sample_list)

        filename = tmp_path / "samples.csv"
        samples.write_table(filename=filename)

        with open(filename) as f:
            rows = list(csv.reader(f))

        assert len(rows) == 101
        headers = [h.strip() for h in rows[0]]
        assert headers == model.joined_paths + [
            "log_likelihood",
            "log_prior",
            "log_posterior",
            "weight",
        ]

        # Every written weight stays above the output.yaml
        # samples_weight_threshold (1e-10), so threshold-applying loads
        # keep the full set (cf. PyAutoFit#1375).
        weight_index = headers.index("weight")
        weights = [float(row[weight_index]) for row in rows[1:]]
        assert min(weights) > 1.0e-10

    def test__functional_at_fifty_thousand(self, monkeypatch):
        monkeypatch.setenv("PYAUTO_TEST_MODE_SAMPLES", "50000")

        model = af.Model(af.m.MockClassx2)

        sample_list = af.DynestyStatic._build_fake_samples(
            model=model,
            parameter_vector=[1.0, 2.0],
            log_likelihood=-10.0,
        )

        assert len(sample_list) == 50000
        assert sample_list[0].parameter_lists_for_model(model) == [1.0, 2.0]

        weights = np.array([sample.weight for sample in sample_list])
        assert weights.sum() == pytest.approx(1.0)
        assert weights.min() > 1.0e-10


class TestUpdaterPathsRefresh:
    """
    The cached ``SearchUpdater`` must be invalidated when ``self.paths``
    is reassigned to a new object — otherwise output (samples, viz,
    profiling) keeps writing under the FIRST paths the search ever saw.
    The EP loop does this routinely:
    ``AbstractSearch.optimise(factor_approx)`` reassigns ``self.paths``
    to a fresh ``SubDirectoryPaths`` per factor and per EP iteration.
    """

    def test__updater_refreshes_when_paths_reassigned(self):
        search = af.DynestyStatic(name="updater_paths_test_a")
        first_updater = search._updater
        first_paths = search.paths
        # Sanity: cache hit on second access with no path change.
        assert search._updater is first_updater

        search.paths = af.DirectoryPaths(name="updater_paths_test_b")
        second_updater = search._updater

        assert second_updater is not first_updater, (
            "Expected a fresh SearchUpdater after self.paths was reassigned"
        )
        assert second_updater._paths is search.paths
        assert second_updater._paths is not first_paths


class TestLabels:
    def test_param_names(self):
        model = af.Model(af.m.MockClassx4)
        assert [
            "one",
            "two",
            "three",
            "four",
        ] == model.model_component_and_parameter_names

    def test_label_config(self):
        assert conf.instance["notation"]["label"]["label"]["one"] == "one_label"
        assert conf.instance["notation"]["label"]["label"]["two"] == "two_label"
        assert conf.instance["notation"]["label"]["label"]["three"] == "three_label"
        assert conf.instance["notation"]["label"]["label"]["four"] == "four_label"


class TestBypassWritesCompleted:
    """
    A bypassed fit (PYAUTO_TEST_MODE=2/3) must mark itself complete, exactly
    as start_resume_fit does — otherwise paths.is_complete stays False and a
    rerun re-bypasses the whole pipeline instead of resuming from the
    completed output (found by the SLaM resume profiler, autolens_profiling#70).
    """

    def test__bypass_marks_complete__second_fit_takes_completed_path(
        self, monkeypatch
    ):
        monkeypatch.setenv("PYAUTO_TEST_MODE", "3")

        unique_tag = "bypass_completed_test"

        model = af.Model(af.m.MockClassx2)
        analysis = af.m.MockAnalysis()

        search = af.DynestyStatic(name="bypass_completed", unique_tag=unique_tag)
        search.fit(model=model, analysis=analysis)

        # Under the test config's `remove_files: true` only the zip remains
        # after fit() — the marker must have been zipped up with the output.
        import zipfile
        from pathlib import Path

        with zipfile.ZipFile(search.paths._zip_path) as f:
            assert ".completed" in {Path(n).name for n in f.namelist()}

        search_resumed = af.DynestyStatic(
            name="bypass_completed", unique_tag=unique_tag
        )

        def _poison(*args, **kwargs):
            raise AssertionError(
                "start_resume_fit taken — the .completed marker is missing"
            )

        monkeypatch.setattr(search_resumed, "start_resume_fit", _poison)

        result = search_resumed.fit(
            model=af.Model(af.m.MockClassx2), analysis=af.m.MockAnalysis()
        )

        assert result is not None
