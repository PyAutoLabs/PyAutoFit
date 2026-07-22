import numpy as np
import pytest

from autofit.database import JSON
from autofit import database as db
from astropy.io import fits


@pytest.fixture(name="fit")
def make_fit():
    return db.Fit()


def test_json():
    json = JSON(name="test", dict={"a": 1})
    assert json.name == "test"
    assert json.dict == {"a": 1}


def test_set_json(fit):
    fit.set_json("test", {"a": 1})
    assert fit.get_json("test") == {"a": 1}


def test_array():
    csv = db.Array(
        name="test",
        array=np.array([[1, 2], [3, 4]]),
    )
    assert csv.name == "test"
    assert (csv.array == [[1, 2], [3, 4]]).all()


def test_set_array(fit):
    fit.set_array("test", np.array([[1, 2], [3, 4]]))
    assert (fit.get_array("test") == [[1, 2], [3, 4]]).all()


@pytest.fixture(name="hdu_array")
def make_hdu_array():
    return np.array([[3, 3], [3, 3]], dtype=np.dtype(">f8"))


@pytest.fixture(name="hdu")
def make_hdu(hdu_array):
    new_hdr = fits.Header()
    return fits.PrimaryHDU(hdu_array, new_hdr)


def test_hdu(hdu, hdu_array):
    db_hdu = db.HDU(name="test", hdu=hdu)
    assert db_hdu.name == "test"

    loaded = db_hdu.hdu
    assert (loaded.data == hdu_array).all()
    assert loaded.header == hdu.header


def test_hdu_without_data():
    """
    A data-less HDU must round-trip. The first HDU of a multi-extension FITS is
    conventionally an empty `PrimaryHDU` — `AggregateFITS` emits exactly that —
    so dereferencing `hdu.data.dtype` here broke every such scrape.
    """
    db_hdu = db.HDU(name="test", hdu=fits.PrimaryHDU())

    assert not db_hdu.has_data

    loaded = db_hdu.hdu
    assert isinstance(loaded, fits.PrimaryHDU)
    assert loaded.data is None


def test_set_fits_with_empty_primary_hdu(fit, hdu_array):
    """
    The shape the database scrape actually meets: an empty `PrimaryHDU`
    followed by named image extensions.
    """
    image_hdu = fits.ImageHDU(hdu_array)
    image_hdu.header["EXTNAME"] = "MODEL_IMAGE"

    fit.set_fits("test", fits.HDUList([fits.PrimaryHDU(), image_hdu]))

    loaded = fit.get_fits("test")
    assert len(loaded) == 2
    assert loaded[0].data is None
    assert (loaded[1].data == hdu_array).all()
