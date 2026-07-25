import os
from os import path
import numpy as np
from typing import Optional

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