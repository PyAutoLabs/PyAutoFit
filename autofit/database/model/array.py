import numpy as np

from autonerves.class_path import get_class_path, get_class
from .model import Object
from ..sqlalchemy_ import sa


class Array(Object):
    """
    A serialised numpy array
    """

    __tablename__ = "array"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    id = sa.Column(
        sa.Integer,
        sa.ForeignKey("object.id"),
        primary_key=True,
        index=True,
    )

    __mapper_args__ = {"polymorphic_identity": "array"}

    bytes = sa.Column(sa.LargeBinary)
    _dtype = sa.Column(sa.String)
    _shape = sa.Column(sa.String)

    fit_id = sa.Column(sa.String, sa.ForeignKey("fit.id"))
    fit = sa.orm.relationship(
        "Fit",
        uselist=False,
        foreign_keys=[fit_id],
        back_populates="arrays",
    )

    @property
    def shape(self):
        return tuple(map(int, self._shape.split(",")))

    @shape.setter
    def shape(self, shape):
        self._shape = ",".join(map(str, shape))

    @property
    def array(self):
        return np.frombuffer(
            self.bytes,
            dtype=self.dtype,
        ).reshape(self.shape)

    @array.setter
    def array(self, array):
        self._dtype = get_class_path(getattr(np, array.dtype.name))
        self.shape = array.shape
        self.bytes = array.tobytes()

    @property
    def dtype(self):
        return get_class(self._dtype)

    @property
    def value(self):
        return self.array

    def __call__(self, *args, **kwargs):
        return self.value


class Fits(Object):
    """
    A serialised astropy.io.fits.HDUList
    """

    __tablename__ = "fits"

    id = sa.Column(
        sa.Integer,
        sa.ForeignKey("object.id"),
        primary_key=True,
        index=True,
    )

    __mapper_args__ = {
        "polymorphic_identity": "fits",
    }

    hdus = sa.orm.relationship(
        "HDU",
        back_populates="fits",
        foreign_keys="HDU.fits_id",
    )

    fit_id = sa.Column(sa.String, sa.ForeignKey("fit.id"))
    fit = sa.orm.relationship(
        "Fit",
        uselist=False,
        foreign_keys=[fit_id],
        back_populates="fits",
    )

    @property
    def hdu_list(self):
        from astropy.io import fits

        return fits.HDUList([hdu.hdu for hdu in self.hdus])

    @hdu_list.setter
    def hdu_list(self, hdu_list):
        self.hdus = [HDU(hdu=hdu) for hdu in hdu_list]

    @property
    def value(self):
        return self.hdu_list


class HDU(Array):
    """
    A serialised astropy.io.fits.PrimaryHDU
    """

    __tablename__ = "hdu"

    id = sa.Column(
        sa.Integer,
        sa.ForeignKey("array.id"),
        primary_key=True,
        index=True,
    )

    _header = sa.Column(sa.String)

    __mapper_args__ = {
        "polymorphic_identity": "hdu",
    }

    is_primary = sa.Column(sa.Boolean)

    fit = sa.orm.relationship(
        "Fit",
        uselist=False,
        foreign_keys=[Array.fit_id],
        back_populates="hdus",
    )

    fits_id = sa.Column(
        sa.Integer,
        sa.ForeignKey("fits.id"),
    )
    fits = sa.orm.relationship(
        "Fits",
        uselist=False,
        foreign_keys=[fits_id],
        back_populates="hdus",
    )

    @property
    def header(self):
        """
        The header of the HDU
        """
        from astropy.io.fits import Header

        return Header.fromstring(self._header)

    @header.setter
    def header(self, header):
        self._header = header.tostring()

    @property
    def has_data(self):
        """
        Whether this HDU carries an array payload. A data-less HDU stores no
        shape, which is how it is distinguished from one holding an array.
        """
        return self._shape is not None

    @property
    def hdu(self):
        from astropy.io import fits

        type_ = fits.PrimaryHDU if self.is_primary else fits.ImageHDU

        return type_(
            self.array if self.has_data else None,
            self.header,
        )

    @hdu.setter
    def hdu(self, hdu):
        from astropy.io import fits

        self.is_primary = isinstance(hdu, fits.PrimaryHDU)

        # A data-less HDU is legitimate and routine: the first HDU of a
        # multi-extension FITS is conventionally an empty PrimaryHDU, and
        # `AggregateFITS` emits exactly that. Leave the array columns null
        # rather than dereferencing `hdu.data.dtype` on None.
        if hdu.data is None:
            self._dtype = None
            self._shape = None
            self.bytes = None
        else:
            self.array = hdu.data

        self.header = hdu.header

    @property
    def value(self):
        return self.hdu

    @property
    def dtype(self):
        return np.dtype(">f8")
