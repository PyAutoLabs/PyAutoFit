import json

import numpy as np
import pytest

import autofit as af
from autofit.tools.util import open_atomic


class TestOpenAtomic:
    def test__successful_write_replaces_the_file(self, tmp_path):
        path = tmp_path / "f.json"
        path.write_text('{"old": 1}')

        with open_atomic(path) as f:
            json.dump({"new": 2}, f)

        assert json.loads(path.read_text()) == {"new": 2}

    def test__failed_write_leaves_the_original_intact(self, tmp_path):
        """
        The whole point. A plain ``open(path, "w+")`` truncates first, so a
        failure partway through destroys the previous file.
        """
        path = tmp_path / "f.json"
        path.write_text('{"old": 1}')

        with pytest.raises(TypeError):
            with open_atomic(path) as f:
                f.write('{"partial": ')
                raise TypeError("Object of type float32 is not JSON serializable")

        assert json.loads(path.read_text()) == {"old": 1}

    def test__failed_write_leaves_no_temp_file_behind(self, tmp_path):
        path = tmp_path / "f.json"
        path.write_text('{"old": 1}')

        with pytest.raises(TypeError):
            with open_atomic(path) as f:
                f.write("junk")
                raise TypeError

        assert [p.name for p in tmp_path.iterdir()] == ["f.json"]

    def test__keyboard_interrupt_also_cleans_up(self, tmp_path):
        """
        ``BaseException``, not ``Exception``: an interrupt mid-write leaves the
        same debris and must not leak a .tmp file either.
        """
        path = tmp_path / "f.json"
        path.write_text('{"old": 1}')

        with pytest.raises(KeyboardInterrupt):
            with open_atomic(path) as f:
                f.write("junk")
                raise KeyboardInterrupt

        assert json.loads(path.read_text()) == {"old": 1}
        assert [p.name for p in tmp_path.iterdir()] == ["f.json"]

    def test__creates_a_file_that_did_not_exist(self, tmp_path):
        path = tmp_path / "sub" / "f.json"

        with open_atomic(path) as f:
            json.dump({"a": 1}, f)

        assert json.loads(path.read_text()) == {"a": 1}

    def test__binary_mode(self, tmp_path):
        """``save_search_internal`` writes dill, so binary must work too."""
        path = tmp_path / "f.bin"

        with open_atomic(path, "wb") as f:
            f.write(b"\x00\x01")

        assert path.read_bytes() == b"\x00\x01"


class TestSaveJsonIsAtomic:
    def test__a_failed_save_json_does_not_destroy_the_previous_file(
        self, output_directory
    ):
        """
        The field sequence: a run writes a good summary, a later run dies while
        rewriting it, and the half-written file poisons every run after that.
        """
        paths = af.DirectoryPaths(name="atomic_save_json")
        paths._identifier = "id"

        paths.save_json(name="counters", object_dict={"clipped": 4})
        assert paths.load_json(name="counters") == {"clipped": 4}

        class Unserialisable:
            pass

        with pytest.raises(TypeError):
            paths.save_json(
                name="counters", object_dict={"clipped": Unserialisable()}
            )

        # Still readable, and still the OLD value -- not truncated.
        assert paths.load_json(name="counters") == {"clipped": 4}


class TestCorruptOutputDoesNotPoisonTheNextRun:
    def test__truncated_summary_lets_the_next_run_proceed(
        self, output_directory
    ):
        """
        End-to-end, because this bug is only visible across two runs.

        Before the fix the second run died with an opaque
        ``JSONDecodeError: Expecting value: line 1 column 13`` raised from
        inside an OPTIONAL likelihood sanity check, and kept dying on every
        rerun of the same name.
        """
        from autofit import example

        xvalues = np.arange(60)
        truth = example.Gaussian(centre=30.0, normalization=25.0, sigma=8.0)
        data = np.asarray(truth.model_data_from(xvalues=xvalues))
        noise_map = np.full(60, 1.0)

        def analysis():
            instance = example.Analysis(data=data, noise_map=noise_map)
            instance._use_jax = False
            return instance

        def model():
            built = af.Model(example.Gaussian)
            built.centre = af.UniformPrior(lower_limit=5.0, upper_limit=35.0)
            built.normalization = af.UniformPrior(
                lower_limit=10.0, upper_limit=40.0
            )
            built.sigma = af.UniformPrior(lower_limit=1.0, upper_limit=20.0)
            return built

        name = "corrupt_resume"

        search = af.LBFGS(name=name, maxiter=10)
        search.fit(model=model(), analysis=analysis())

        out = search.paths.output_path

        # The test config prunes output files after a run, so write the
        # corrupt state directly rather than depending on what survived. This
        # is the same half-written file a truncating "w+" leaves behind.
        summary = out / "files" / "samples_summary.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text('{"partial": ')

        # Not `.completed` by rglob -- the marker's location is the paths
        # object's business, and an interrupted run never wrote it anyway.
        search.paths._has_completed_path.unlink(missing_ok=True)

        resumed = af.LBFGS(name=name, maxiter=10)
        assert not resumed.paths.is_complete, (
            "the second run must take the resume path, not be short-circuited "
            "as already-complete -- otherwise this asserts nothing"
        )

        # The regression: before the fix this raised JSONDecodeError out of an
        # optional sanity check, and did so on every rerun of the same name.
        result = resumed.fit(model=model(), analysis=analysis())

        assert result is not None

    def test__json_decode_error_is_a_value_error(self):
        """
        The crux of why this was missed. ``JSONDecodeError`` is neither a
        ``FileNotFoundError``, a ``TypeError`` nor a ``KeyError``, so it fell
        through every guard on the resume path.
        """
        assert issubclass(json.JSONDecodeError, ValueError)
        assert not issubclass(json.JSONDecodeError, (TypeError, KeyError))
