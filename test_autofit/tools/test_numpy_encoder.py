import json

import numpy as np
import pytest

import autofit as af
from autofit.tools.util import NumpyEncoder


class TestNumpyEncoder:
    def test__float32_round_trips_rather_than_raising(self):
        """
        The bug this closes. ``float32`` does not subclass Python ``float``, so
        a bare ``json.dump`` raises ``TypeError`` on it.
        """
        with pytest.raises(TypeError):
            json.dumps({"x": np.float32(1.5)})

        assert json.loads(json.dumps({"x": np.float32(1.5)}, cls=NumpyEncoder)) == {
            "x": 1.5
        }

    @pytest.mark.parametrize(
        "value, expected, expected_type",
        [
            (np.float32(1.5), 1.5, float),
            (np.float16(0.5), 0.5, float),
            (np.int32(3), 3, int),
            (np.int64(4), 4, int),
            (np.uint8(5), 5, int),
            (np.bool_(True), True, bool),
        ],
    )
    def test__numpy_scalars_become_plain_python(
        self, value, expected, expected_type
    ):
        loaded = json.loads(json.dumps({"x": value}, cls=NumpyEncoder))["x"]

        assert loaded == expected
        assert type(loaded) is expected_type

    def test__arrays_become_lists(self):
        loaded = json.loads(
            json.dumps({"x": np.array([[1, 2], [3, 4]], dtype=np.int32)},
                       cls=NumpyEncoder)
        )

        assert loaded == {"x": [[1, 2], [3, 4]]}

    def test__float64_is_unchanged(self):
        """
        ``float64`` already worked, because it subclasses Python ``float``. The
        encoder must not alter it -- and must not lose precision routing it.
        """
        value = np.float64(0.1234567890123456789)

        assert json.dumps({"x": value}) == json.dumps({"x": value}, cls=NumpyEncoder)
        assert json.loads(json.dumps({"x": value}, cls=NumpyEncoder))["x"] == float(
            value
        )

    def test__float32_precision_is_not_invented(self):
        """
        ``.item()`` widens float32 to a Python double. The value must be the
        float32's exact value, not a re-rounded decimal.
        """
        loaded = json.loads(json.dumps({"x": np.float32(0.1)}, cls=NumpyEncoder))["x"]

        assert loaded == float(np.float32(0.1))

    def test__unserialisable_objects_still_raise(self):
        """
        The encoder widens what can be written; it must not silently swallow a
        genuinely unserialisable object.
        """
        with pytest.raises(TypeError):
            json.dumps({"x": object()}, cls=NumpyEncoder)


class TestOutputPathsUseTheEncoder:
    def test__save_json_writes_a_float32(self, output_directory):
        """
        ``DirectoryPaths.save_json`` is where this fired in the field -- at the
        END of a successful fit, discarding the whole run at its output step.
        """
        paths = af.DirectoryPaths(name="save_json_float32")
        paths._identifier = "id"

        paths.save_json(name="counters", object_dict={"clipped": np.float32(4.0)})

        assert paths.load_json(name="counters") == {"clipped": 4.0}

    def test__samples_info_json_writes_numpy_scalars(self, output_directory, tmp_path):
        """
        ``samples_info`` is the search's own diagnostic channel, so it is the
        dict most likely to carry a NumPy scalar out of a search's internals.
        """
        from autofit import example

        model = af.Model(example.Gaussian)

        samples = af.Samples(
            model=model,
            sample_list=af.Sample.from_lists(
                model=model,
                parameter_lists=[[1.0, 2.0, 3.0]],
                log_likelihood_list=[1.0],
                log_prior_list=[0.0],
                weight_list=[1.0],
            ),
            samples_info={
                "n_clipped_lane_steps": np.int32(414),
                "best_fom": np.float32(-2.5),
            },
        )

        filename = tmp_path / "info.json"
        samples.info_to_json(filename=filename)

        with open(filename) as f:
            loaded = json.load(f)

        # ``samples_info`` also carries an auto-added ``class_path``, so assert
        # on the values under test rather than on the whole dict.
        assert loaded["n_clipped_lane_steps"] == 414
        assert type(loaded["n_clipped_lane_steps"]) is int
        assert loaded["best_fom"] == -2.5
        assert type(loaded["best_fom"]) is float
