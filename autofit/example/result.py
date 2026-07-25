import numpy as np

import autofit as af

class ResultExample(af.Result):

    @property
    def max_log_likelihood_model_data_1d(self) -> np.ndarray:
        """
        Returns the maximum log likelihood model's 1D model data.

        This is an example of how we can pass the `Analysis` class a custom `Result` object and extend this result
        object with new properties that are specific to the model-fit we are performing.
        """
        return self.analysis.model_data_1d_from(instance=self.instance)
