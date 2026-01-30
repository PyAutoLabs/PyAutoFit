from typing import Optional, List

from autofit.database.sqlalchemy_ import sa


class SettingsSearch:
    def __init__(
        self,
        path_prefix: str,
        unique_tag: Optional[str] = None,
        number_of_cores: Optional[int] = 1,
        session: Optional[sa.orm.Session] = None,
        info: Optional[dict] = None,
        use_jax_vmap: bool = True,
    ):
        """
        Stores all the input settings that are used in search's and their `fit functions.

        This is used for more concisely passing settings through pipelines written using the empirical Bayesian
        functionality.

        Parameters
        ----------
        path_prefix
            The prefix of folders between the output path and the search folders.
        unique_tag
            The unique tag for this model-fit, which will be given a unique entry in the sqlite database and also acts as
            the folder after the path prefix and before the search name. This is typically the name of the dataset.
        number_of_cores
            The number of CPU cores used to parallelize the model-fit. This is used internally in a non-linear search
            for most model fits, but is done on a per-fit basis for grid based searches (e.g. sensitivity mapping).
        session
            The SQLite database session which is active means results are directly wrtten to the SQLite database
            at the end of a fit and loaded from the database at the start.
        info
            Optional dictionary containing information about the model-fit that is stored in the database and can be
            loaded by the aggregator after the model-fit is complete.
        """

        self.path_prefix = path_prefix
        self.unique_tag = unique_tag
        self.number_of_cores = number_of_cores
        self.session = session
        self.use_jax_vmap = use_jax_vmap

        self.info = info

    @property
    def search_dict(self):
        return {
            "path_prefix": self.path_prefix,
            "unique_tag": self.unique_tag,
            "number_of_cores": self.number_of_cores,
            "session": self.session,
            "use_jax_vmap": self.use_jax_vmap,
        }

    @property
    def fit_dict(self):
        return {"info": self.info}
