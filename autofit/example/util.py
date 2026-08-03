import json
import os
from os import path
import numpy as np
from typing import List, Optional

from autonerves.dictable import to_dict

import autofit as af

def plot_profile_1d(
    xvalues : np.ndarray,
    profile_1d: np.ndarray,
    title:Optional[str]=None,
    ylabel:Optional[str]=None,
    errors:Optional[np.ndarray]=None,
    color:Optional[str]="k",
    output_path:Optional[str]=None,
    output_filename:Optional[str]=None,
):
    """
    Plot a 1D image of data on a plot of x versus y, where the x-axis is the x coordinate of the 1D profile
    and the y-axis is the value of the 1D profile at that coordinate.

    The function include options to output the image to the hard-disk as a .png.

    Parameters
    ----------
    xvalues
        The x-coordinates the profile is defined on.
    profile_1d
        The normalization values of the profile which are plotted.
    ylabel
        The y-label of the plot.
    errors
        The errors on each data point, which are related to its noise-map.
    output_path
        The path the image is to be output to hard-disk as a .png.
    output_filename
        The filename of the file if it is output as a .png.
    output_format
        Determines where the plot is displayed on your screen ("show") or output to the hard-disk as a png ("png").
    """
    import matplotlib.pyplot as plt

    plt.errorbar(
        x=xvalues, y=profile_1d, yerr=errors, color=color, ecolor="k", elinewidth=1, capsize=2
    )
    plt.title(title)
    plt.xlabel("x value of profile")
    plt.ylabel(ylabel)
    if output_filename is None:
        plt.show()
    else:
        if not path.exists(output_path):
            os.makedirs(output_path)
        plt.savefig(output_path / f"{output_filename}.png")
    plt.clf()
    plt.close()


"""
The functions below simulate the example 1D datasets fitted throughout the PyAutoFit workspaces.

They live in the library rather than beside the simulator scripts because a Jupyter notebook kernel
has no script directory on `sys.path` — a local `import util` resolves for `python
scripts/simulators/simulators.py` but can never resolve in the generated notebook.
"""

PIXELS = 100
SIGNAL_TO_NOISE_RATIO = 25.0


def _kernel_1d() -> np.ndarray:
    """
    Return the normalized 1D Gaussian kernel that the blurred datasets are convolved with.
    """
    kernel_pixels = 21
    kernel_sigma = 5.0
    kernel_centre = 10.0

    kernel_xvalues = np.subtract(np.arange(kernel_pixels), kernel_centre)

    kernel = np.multiply(
        np.divide(1.0, kernel_sigma * np.sqrt(2.0 * np.pi)),
        np.exp(-0.5 * np.square(np.divide(kernel_xvalues, kernel_sigma))),
    )

    return kernel / np.sum(kernel)


def _noise_map_1d() -> np.ndarray:
    """
    Return the noise-map, which is required when evaluating the chi-squared value of the likelihood.
    """
    return (1.0 / SIGNAL_TO_NOISE_RATIO) * np.ones(PIXELS)


def _output_dataset(
    dataset_path: str,
    data: np.ndarray,
    noise_map: np.ndarray,
    kernel: Optional[np.ndarray] = None,
):
    """
    Output the data, noise-map and (optionally) the convolution kernel to the `dataset` folder, so
    they can be loaded and used in other example scripts.

    Parameters
    ----------
    dataset_path
        The folder the dataset .json files are output to.
    data
        The simulated data that is fitted.
    noise_map
        The noise-map of the simulated data.
    kernel
        The convolution kernel, output only for the blurred datasets.
    """
    af.util.numpy_array_to_json(
        array=data, file_path=path.join(dataset_path, "data.json"), overwrite=True
    )
    af.util.numpy_array_to_json(
        array=noise_map,
        file_path=path.join(dataset_path, "noise_map.json"),
        overwrite=True,
    )

    if kernel is not None:
        af.util.numpy_array_to_json(
            array=kernel,
            file_path=path.join(dataset_path, "kernel.json"),
            overwrite=True,
        )


def _output_model_json(
    dataset_path: str,
    profile_1d_list: List,
    indexed: bool,
    strict: bool = False,
):
    """
    Output each profile to a .json file so its parameters can be referred to in the future.

    Parameters
    ----------
    dataset_path
        The folder the model .json files are output to.
    profile_1d_list
        The profiles the dataset was simulated from.
    indexed
        If `True` the profiles are written as `model_{i}.json`, else the single profile is written
        as `model.json`. Set by the caller rather than inferred from the list length, so a
        one-profile list keeps the indexed filenames its dataset folder is read with.
    strict
        If `False` a profile that cannot be serialized is skipped rather than raising.
    """
    for i, profile in enumerate(profile_1d_list):
        filename = f"model_{i}.json" if indexed else "model.json"

        with open(path.join(dataset_path, filename), "w+") as f:
            if strict:
                json.dump(to_dict(profile), f, indent=4)
            else:
                try:
                    json.dump(to_dict(profile), f, indent=4)
                except (TypeError, ValueError):
                    pass


def _output_figure(
    dataset_path: str,
    xvalues: np.ndarray,
    data: np.ndarray,
    noise_map: np.ndarray,
    title: str,
    model_data_1d: Optional[np.ndarray] = None,
    model_data_1d_list: Optional[List[np.ndarray]] = None,
):
    """
    Output a .png of the simulated dataset, optionally overlaying the model profiles it was
    simulated from.

    Parameters
    ----------
    dataset_path
        The folder the `image.png` is output to.
    xvalues
        The x-coordinates the profile is defined on.
    data
        The simulated data that is fitted.
    noise_map
        The noise-map of the simulated data, plotted as the error bars.
    title
        The title of the figure.
    model_data_1d
        The summed model profile, overlaid in red if input.
    model_data_1d_list
        The individual model profiles, overlaid as dashed lines if input.
    """
    import matplotlib.pyplot as plt

    plt.errorbar(
        x=xvalues,
        y=data,
        yerr=noise_map,
        linestyle="",
        color="k",
        ecolor="k",
        elinewidth=1,
        capsize=2,
    )

    if model_data_1d is not None:
        plt.plot(range(data.shape[0]), model_data_1d, color="r")

    if model_data_1d_list is not None:
        for model_data_1d_individual in model_data_1d_list:
            plt.plot(range(data.shape[0]), model_data_1d_individual, "--")

    plt.title(title)
    plt.xlabel("x values of profile")
    plt.ylabel("Profile normalization")
    plt.savefig(path.join(dataset_path, "image.png"))
    plt.close()


def simulate_dataset_1d_via_gaussian_from(gaussian, dataset_path: str):
    """
    Simulate a 1D dataset from a single `Gaussian` and output it to the `dataset` folder.

    The `Gaussian` is evaluated on `PIXELS` xvalues to create its model profile, noise is added at
    a fixed signal-to-noise ratio, and the data, noise-map, an `image.png` and the model .json are
    written to `dataset_path`.

    Parameters
    ----------
    gaussian
        The `Gaussian` profile the dataset is simulated from.
    dataset_path
        The folder the simulated dataset is output to.
    """
    xvalues = np.arange(PIXELS)

    model_data_1d = gaussian.model_data_from(xvalues=xvalues)

    noise = np.random.normal(0.0, 1.0 / SIGNAL_TO_NOISE_RATIO, PIXELS)

    data = model_data_1d + noise
    noise_map = _noise_map_1d()

    _output_dataset(dataset_path=dataset_path, data=data, noise_map=noise_map)

    _output_figure(
        dataset_path=dataset_path,
        xvalues=xvalues,
        data=data,
        noise_map=noise_map,
        title="1D Gaussian Dataset.",
    )

    _output_model_json(
        dataset_path=dataset_path, profile_1d_list=[gaussian], indexed=False
    )


def simulate_data_1d_with_kernel_via_gaussian_from(gaussian, dataset_path: str):
    """
    Simulate a 1D dataset from a single `Gaussian` convolved with a Gaussian kernel, and output it
    to the `dataset` folder.

    Identical to `simulate_dataset_1d_via_gaussian_from` except the model profile is convolved
    before the noise is added, and the kernel is output alongside the data so the fit can deconvolve
    it.

    Parameters
    ----------
    gaussian
        The `Gaussian` profile the dataset is simulated from.
    dataset_path
        The folder the simulated dataset is output to.
    """
    xvalues = np.arange(PIXELS)

    model_data_1d = gaussian.model_data_from(xvalues=xvalues)

    kernel = _kernel_1d()

    blurred_model_data_1d = np.convolve(model_data_1d, kernel, mode="same")

    noise = np.random.normal(0.0, 1.0 / SIGNAL_TO_NOISE_RATIO, PIXELS)

    data = blurred_model_data_1d + noise
    noise_map = _noise_map_1d()

    _output_dataset(
        dataset_path=dataset_path, data=data, noise_map=noise_map, kernel=kernel
    )

    _output_figure(
        dataset_path=dataset_path,
        xvalues=xvalues,
        data=data,
        noise_map=noise_map,
        title="1D Gaussian Dataset with Convolver Blurring.",
    )

    _output_model_json(
        dataset_path=dataset_path,
        profile_1d_list=[gaussian],
        indexed=False,
        strict=True,
    )


def simulate_dataset_1d_via_profile_1d_list_from(profile_1d_list: List, dataset_path: str):
    """
    Simulate a 1D dataset from a list of profiles and output it to the `dataset` folder.

    Every profile is evaluated on `PIXELS` xvalues and summed to create the overall model profile.
    The maximum log likelihood of the dataset (that of the profiles it was simulated from) is
    output alongside it as `max_log_likelihood.json`.

    Parameters
    ----------
    profile_1d_list
        The profiles which are summed to simulate the dataset.
    dataset_path
        The folder the simulated dataset is output to.
    """
    xvalues = np.arange(PIXELS)

    model_data_1d_list = [
        profile_1d.model_data_from(xvalues=xvalues) for profile_1d in profile_1d_list
    ]

    model_data_1d = sum(model_data_1d_list)

    noise = np.random.normal(0.0, 1.0 / SIGNAL_TO_NOISE_RATIO, PIXELS)

    data = model_data_1d + noise
    noise_map = _noise_map_1d()

    _output_dataset(dataset_path=dataset_path, data=data, noise_map=noise_map)

    _output_figure(
        dataset_path=dataset_path,
        xvalues=xvalues,
        data=data,
        noise_map=noise_map,
        title="1D Profiles Dataset.",
        model_data_1d=model_data_1d,
        model_data_1d_list=model_data_1d_list,
    )

    _output_model_json(
        dataset_path=dataset_path, profile_1d_list=profile_1d_list, indexed=True
    )

    chi_squared = np.sum(((data - model_data_1d) / noise_map) ** 2)
    noise_normalization = np.sum(np.log(2 * np.pi * noise_map**2.0))
    log_likelihood = -0.5 * (chi_squared + noise_normalization)

    with open(path.join(dataset_path, "max_log_likelihood.json"), "w+") as f:
        json.dump({"log_likelihood": log_likelihood}, f, indent=4)


def simulate_data_1d_with_kernel_via_profile_1d_list_from(
    profile_1d_list: List, dataset_path: str
):
    """
    Simulate a 1D dataset from a list of profiles convolved with a Gaussian kernel, and output it
    to the `dataset` folder.

    Identical to `simulate_dataset_1d_via_profile_1d_list_from` except the summed model profile is
    convolved before the noise is added, and the kernel is output alongside the data.

    Parameters
    ----------
    profile_1d_list
        The profiles which are summed to simulate the dataset.
    dataset_path
        The folder the simulated dataset is output to.
    """
    xvalues = np.arange(PIXELS)

    model_data_1d = np.zeros(shape=PIXELS)

    for profile in profile_1d_list:
        model_data_1d += profile.model_data_from(xvalues=xvalues)

    kernel = _kernel_1d()

    blurred_model_data_1d = np.convolve(model_data_1d, kernel, mode="same")

    noise = np.random.normal(0.0, 1.0 / SIGNAL_TO_NOISE_RATIO, PIXELS)

    data = blurred_model_data_1d + noise
    noise_map = _noise_map_1d()

    _output_dataset(
        dataset_path=dataset_path, data=data, noise_map=noise_map, kernel=kernel
    )

    _output_figure(
        dataset_path=dataset_path,
        xvalues=xvalues,
        data=data,
        noise_map=noise_map,
        title="1D Profiles Dataset with Convolver Blurring.",
    )

    _output_model_json(
        dataset_path=dataset_path,
        profile_1d_list=profile_1d_list,
        indexed=True,
        strict=True,
    )