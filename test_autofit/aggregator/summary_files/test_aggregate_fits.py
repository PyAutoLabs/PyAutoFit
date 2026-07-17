from enum import Enum
import pytest

import autofit as af
from pathlib import Path


class FITSFit(Enum):
    """
    The HDUs that can be extracted from the fit.fits file.
    """

    ModelData = "MODEL_IMAGE"
    ResidualMap = "RESIDUAL_MAP"
    NormalizedResidualMap = "NORMALIZED_RESIDUAL_MAP"
    ChiSquaredMap = "CHI_SQUARED_MAP"


@pytest.fixture(name="summary")
def make_summary(aggregator):
    return af.AggregateFITS(aggregator)


def test_aggregate(summary):
    result = summary.extract_fits(
        [
            FITSFit.ModelData,
            FITSFit.ResidualMap,
        ],
    )
    assert len(result) == 5


def test_output_to_file(summary, output_directory):
    folder = output_directory / "fits"
    summary.output_to_folder(
        folder,
        name="id",
        hdus=[
            FITSFit.ModelData,
            FITSFit.ResidualMap,
        ],
    )
    assert list(folder.glob("*"))


def test_list_of_names(summary, output_directory):
    summary.output_to_folder(
        output_directory,
        ["one", "two"],
        [
            FITSFit.ModelData,
            FITSFit.ResidualMap,
        ],
    )
    assert set([path.name for path in Path(output_directory).glob("*.fits")]) == set([
        "one.fits",
        "two.fits",
    ])


def test_extract_fits_closes_files(summary):
    """
    Each source fits file is opened once per result and closed deterministically —
    previously one handle leaked per requested HDU, exhausting the open-file limit
    when aggregating many hundreds of results.
    """
    import os

    if not os.path.isdir("/proc/self/fd"):
        pytest.skip("file-descriptor counting requires /proc")

    before = len(os.listdir("/proc/self/fd"))
    summary.extract_fits([FITSFit.ModelData, FITSFit.ResidualMap])
    after = len(os.listdir("/proc/self/fd"))

    assert after <= before + 1
