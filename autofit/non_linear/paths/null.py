import tempfile
from typing import Dict, Optional

from .abstract import AbstractPaths


class NullPaths(AbstractPaths):
    """
    Null version of paths object for avoiding writing of files to disk
    """

    stores_search_internal = False

    @property
    def samples(self):
        return None

    def save_latent_samples(self, latent_samples):
        pass

    def save_json(self, name, object_dict: dict, prefix: str = ""):
        pass

    def load_json(self, name, prefix: str = "") -> dict:
        pass

    def save_array(self, name, array):
        pass

    def load_array(self, name):
        pass

    def save_fits(self, name: str, fits, prefix: str = ""):
        pass

    def load_fits(self, name: str, prefix: str = ""):
        pass

    def __init__(self):
        super().__init__()
        self.objects = dict()
        self._samples_path = tempfile.mkdtemp()

    def save_summary(
        self,
        samples,
        latent_samples,
        log_likelihood_function_time,
        visualization_time = None,
    ):
        pass

    @property
    def samples_path(self) -> str:
        return self._samples_path

    @AbstractPaths.parent.setter
    def parent(self, parent):
        pass

    @property
    def is_grid_search(self) -> bool:
        return False

    def create_child(
        self,
        name: Optional[str] = None,
        path_prefix: Optional[str] = None,
        is_identifier_in_paths: Optional[bool] = None,
        identifier: Optional[str] = None,
    ) -> "AbstractPaths":
        return NullPaths()

    def save_all(self, search_config_dict=None, info=None):
        pass

    def save_object(self, name: str, obj: object, prefix: str = ""):
        self.objects[name] = obj

    def load_object(self, name: str, prefix: str = ""):
        return self.objects[name]

    def remove_object(self, name: str):
        pass

    def is_object(self, name: str) -> bool:
        pass

    @property
    def is_complete(self) -> bool:
        return False

    def completed(self):
        pass

    def save_search_internal(self, obj):
        pass

    def load_search_internal(self):
        pass

    def remove_search_internal(self):
        pass

    @property
    def search_internal_path(self):
        pass

    def load_samples(self):
        pass

    def save_samples(self, samples):
        pass

    def load_samples_info(self):
        pass

    def zip_remove(self):
        pass
