import os
from pathlib import Path
import numpy as np
import autofit as af

test_path = Path(__file__).resolve().parent / "files" / "path"


class TestJson:
    def test__numpy_array_to_json__output_and_load(self):

        if (test_path / "array_out.json").exists():
            os.remove(test_path / "array_out.json")

        arr = np.array([10.0, 30.0, 40.0, 92.0, 19.0, 20.0])

        af.util.numpy_array_to_json(arr, file_path=test_path / "array_out.json")

        array_load = af.util.numpy_array_from_json(
            file_path=test_path / "array_out.json"
        )

        assert (arr == array_load).all()
