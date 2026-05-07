import inspect
import json
import os
import sys
import zipfile
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Union

import numpy as np

from autoconf import conf


def split_paths(func):
    """
    Split string paths if they are passed.

    e.g. "lens.mass.centre" -> ["lens", "mass", "centre"]
    """

    @wraps(func)
    def wrapper(self, paths):
        paths = [path.split(".") if isinstance(path, str) else path for path in paths]
        return func(self, paths)

    return wrapper


class IntervalCounter:
    def __init__(self, interval):
        self.count = 0
        self.interval = interval

    def __call__(self):
        if self.interval == -1:
            return False
        self.count += 1
        return self.count % self.interval == 0


def zip_directory(source_directory, output=None):
    output = output or f"{source_directory}.zip"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as f:
        for root, dirs, files in os.walk(source_directory):
            for file in files:
                f.write(
                    Path(root) / file,
                    Path(root[len(str(source_directory)):]) / file,
                )


def open_(filename, *flags):
    directory = Path(filename)
    os.makedirs(directory.parent, exist_ok=True)
    return open(filename, *flags)


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def numpy_array_to_json(
    array: np.ndarray, file_path: Union[Path, str], overwrite: bool = False
):
    """
    Write a NumPy array to a json file.

    Parameters
    ----------
    array
        The array that is written to json.
    file_path
        The full path of the file that is output, including the file name and `.json` extension.
    overwrite
        If `True` and a file already exists with the input file_path the .json file is overwritten. If
        `False`, an error will be raised.

    Returns
    -------
    None

    Examples
    --------
    array_2d = np.ones((5,5))
    numpy_array_to_json(array_2d=array_2d, file_path='/path/to/file/filename.json', overwrite=True)
    """

    file_dir = Path(file_path).parent

    if not file_dir.exists():
        os.makedirs(file_dir)

    if overwrite and Path(file_path).exists():
        os.remove(file_path)

    with open(file_path, "w+") as f:
        json.dump(array.tolist(), f)


def numpy_array_from_json(file_path: Union[Path, str]):
    """
    Read a 1D NumPy array from a .json file.

    After loading the NumPy array, the array is flipped upside-down using np.flipud. This is so that the structures
    appear the same orientation as .json files loaded in DS9.

    Parameters
    ----------
    file_path
        The full path of the file that is loaded, including the file name and ``.json`` extension.

    Returns
    -------
    ndarray
        The NumPy array that is loaded from the .json file.

    Examples
    --------
    array_2d = numpy_array_from_json(file_path='/path/to/file/filename.json')
    """
    with open(file_path, "r") as f:
        return np.asarray(json.load(f))


def info_whitespace():
    return conf.instance["general"]["output"]["info_whitespace_length"]
