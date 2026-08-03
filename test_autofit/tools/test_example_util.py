import json
from os import path

import numpy as np
import pytest

import autofit as af


@pytest.fixture(autouse=True)
def agg_backend(monkeypatch):
    """
    The simulate functions write an `image.png`, so pin matplotlib to a non-interactive backend.
    """
    import matplotlib

    matplotlib.use("Agg")


@pytest.fixture
def gaussian():
    return af.ex.Gaussian(centre=50.0, normalization=25.0, sigma=10.0)


@pytest.fixture
def exponential():
    return af.ex.Exponential(centre=50.0, normalization=40.0, rate=0.05)


def _assert_dataset_output(dataset_path, filenames):
    for filename in filenames:
        assert path.exists(path.join(str(dataset_path), filename)), filename

    data = af.util.numpy_array_from_json(
        file_path=path.join(str(dataset_path), "data.json")
    )
    noise_map = af.util.numpy_array_from_json(
        file_path=path.join(str(dataset_path), "noise_map.json")
    )

    assert data.shape == (af.ex.util.PIXELS,)
    assert noise_map.shape == (af.ex.util.PIXELS,)
    assert noise_map == pytest.approx(
        (1.0 / af.ex.util.SIGNAL_TO_NOISE_RATIO) * np.ones(af.ex.util.PIXELS), 1.0e-8
    )


def test__simulate_dataset_1d_via_gaussian_from(gaussian, tmp_path):
    af.ex.simulate_dataset_1d_via_gaussian_from(
        gaussian=gaussian, dataset_path=str(tmp_path)
    )

    _assert_dataset_output(
        tmp_path, ["data.json", "noise_map.json", "image.png", "model.json"]
    )


def test__simulate_data_1d_with_kernel_via_gaussian_from(gaussian, tmp_path):
    af.ex.simulate_data_1d_with_kernel_via_gaussian_from(
        gaussian=gaussian, dataset_path=str(tmp_path)
    )

    _assert_dataset_output(
        tmp_path,
        ["data.json", "noise_map.json", "kernel.json", "image.png", "model.json"],
    )

    kernel = af.util.numpy_array_from_json(
        file_path=path.join(str(tmp_path), "kernel.json")
    )

    assert kernel.sum() == pytest.approx(1.0, 1.0e-8)


def test__simulate_dataset_1d_via_profile_1d_list_from(gaussian, exponential, tmp_path):
    af.ex.simulate_dataset_1d_via_profile_1d_list_from(
        profile_1d_list=[gaussian, exponential], dataset_path=str(tmp_path)
    )

    _assert_dataset_output(
        tmp_path,
        [
            "data.json",
            "noise_map.json",
            "image.png",
            "model_0.json",
            "model_1.json",
            "max_log_likelihood.json",
        ],
    )

    with open(path.join(str(tmp_path), "max_log_likelihood.json")) as f:
        assert "log_likelihood" in json.load(f)


def test__simulate_data_1d_with_kernel_via_profile_1d_list_from(
    gaussian, exponential, tmp_path
):
    af.ex.simulate_data_1d_with_kernel_via_profile_1d_list_from(
        profile_1d_list=[gaussian, exponential], dataset_path=str(tmp_path)
    )

    _assert_dataset_output(
        tmp_path,
        [
            "data.json",
            "noise_map.json",
            "kernel.json",
            "image.png",
            "model_0.json",
            "model_1.json",
        ],
    )


def test__single_profile_list_still_writes_indexed_model_json(gaussian, tmp_path):
    """
    The model filename is chosen by the caller, not inferred from the list length, so a
    one-profile list keeps `model_0.json` rather than collapsing to `model.json`.
    """
    af.ex.simulate_dataset_1d_via_profile_1d_list_from(
        profile_1d_list=[gaussian], dataset_path=str(tmp_path)
    )

    assert path.exists(path.join(str(tmp_path), "model_0.json"))
    assert not path.exists(path.join(str(tmp_path), "model.json"))
