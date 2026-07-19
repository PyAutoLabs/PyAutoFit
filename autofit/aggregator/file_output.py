import json
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Union, Optional, Dict

import numpy as np

import dill

from autonerves.dictable import from_dict


class FileOutput(ABC):
    def __new__(cls, name, path: Path):
        suffix = path.suffix
        if suffix == ".pickle":
            return super().__new__(PickleOutput)
        elif suffix == ".json":
            return super().__new__(JSONOutput)
        elif suffix == ".csv":
            return super().__new__(ArrayOutput)
        elif suffix == ".fits":
            return super().__new__(FITSOutput)
        raise ValueError(f"File {path} is not a valid output file")

    def __init__(self, name: str, path: Path):
        """
        An output file from a fit, which is either a pickle, json, csv or fits file.

        Parameters
        ----------
        name
            The name of the output file. This is the path to the file from the root output directory
            separated by '.' and without a suffix. e.g. output/directory/info.json -> directory.info
        path
            The path to the output file
        """
        self.name = name
        self.path = path

    def __str__(self):
        return f"{self.name} ({self.path})"

    def __repr__(self):
        return f"<{self.__class__.__name__} {self}>"

    @property
    @abstractmethod
    def value(self):
        pass


class ArrayOutput(FileOutput):
    @property
    def value(self) -> np.ndarray:
        """
        The array stored in the csv file
        """
        return np.loadtxt(self.path, delimiter=",")


class JSONOutput(FileOutput):
    @property
    def dict(self) -> Union[dict, list]:
        """
        The dictionary stored in the json file
        """
        with open(self.path) as f:
            return json.load(f)

    @property
    def value(self):
        """
        The object represented by the JSON
        """
        return from_dict(self.dict)

    def value_using_reference(self, reference: Optional[Dict] = None):
        """
        The object represented by the JSON
        """
        return from_dict(self.dict, reference=reference)


class PickleOutput(FileOutput):
    @property
    def value(self):
        """
        The object stored in the pickle file
        """
        with open(self.path, "rb") as f:
            return dill.load(f)


class FITSOutput(FileOutput):
    @property
    def value(self):
        """
        The contents of the fits file
        """
        from astropy.io import fits

        return fits.open(self.path)
