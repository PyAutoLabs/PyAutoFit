import abc
import inspect
from sqlalchemy.orm import Mapped, DeclarativeMeta
from typing import List, Tuple, Any, Iterable, Union, ItemsView, Type

import numpy as np

from autonerves.class_path import get_class, get_class_path
from ..sqlalchemy_ import sa
from sqlalchemy.orm import declarative_base


class DeclarativeABCMeta(DeclarativeMeta, abc.ABCMeta):
    """A metaclass combining SQLAlchemy's DeclarativeMeta and ABCMeta."""


Base = declarative_base(metaclass=DeclarativeABCMeta)

_schema_version = 1


class Object(Base):
    __tablename__ = "object"

    type = sa.Column(sa.String)

    id = sa.Column(
        sa.Integer,
        primary_key=True,
        index=True,
    )

    parent_id = sa.Column(
        sa.Integer,
        sa.ForeignKey("object.id"),
        index=True,
    )
    parent = sa.orm.relationship("Object", uselist=False, remote_side=[id])

    samples_for_id = sa.Column(sa.String, sa.ForeignKey("fit.id"))
    samples_for = sa.orm.relationship(
        "Fit",
        uselist=False,
        foreign_keys=[samples_for_id],
        back_populates="_samples",
    )

    latent_samples_for_id = sa.Column(sa.String, sa.ForeignKey("fit.id"))
    latent_samples_for = sa.orm.relationship(
        "Fit",
        uselist=False,
        foreign_keys=[latent_samples_for_id],
        back_populates="_latent_samples",
    )

    children: Mapped[List["Object"]] = sa.orm.relationship(
        "Object",
        uselist=True,
        back_populates="parent",
    )

    def __len__(self):
        return len(self.children)

    name = sa.Column(sa.String)

    __mapper_args__ = {"polymorphic_identity": "object", "polymorphic_on": type}

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.name}>"

    # noinspection PyProtectedMember
    @classmethod
    def from_object(cls, source, name=""):
        """
        Create a database object for an object in a model.

        The specific database class used depends on the type of
        the object.

        If __getstate__ is defined for any instance then that
        dictionary is used in place of the __dict__ when
        serialising.

        Parameters
        ----------
        source
            A model
        name
            The name of the object wrt its parent

        Returns
        -------
        An instance of a concrete child of this class
        """

        from autofit.mapper.prior_model.prior_model import Model
        from autofit.mapper.prior.abstract import Prior
        from autofit.mapper.prior_model.collection import Collection

        from autofit.mapper.prior.arithmetic.compound import Compound

        if source is None or isinstance(
            source,
            (
                np.broadcast,
                abc.ABCMeta,
                np.ufunc,
            ),
        ):
            from .instance import NoneInstance

            instance = NoneInstance()
        elif isinstance(source, np.ndarray):
            from .array import Array

            # noinspection PyArgumentList
            instance = Array(array=source)
        elif isinstance(source, Model):
            from .prior import Model

            instance = Model._from_object(source)
        elif isinstance(source, Prior):
            from .prior import Prior

            instance = Prior._from_object(source)
        elif isinstance(source, (float, int)):
            from .instance import Value

            instance = Value._from_object(source)
        elif isinstance(source, (tuple, list)):
            from .instance import Collection

            instance = Collection._from_object(source)
        elif isinstance(source, dict):
            from .common import Dict

            instance = Dict._from_object(source)
        elif isinstance(source, Collection):
            from .prior import Collection

            instance = Collection._from_object(source)
        elif isinstance(source, str):
            from .instance import StringValue

            instance = StringValue._from_object(source)

        elif isinstance(source, Compound):
            from .compound import Compound

            instance = Compound._from_object(source)
        elif inspect.isfunction(source):
            from .function import Function

            instance = Function._from_object(source)
        else:
            from .instance import Instance

            instance = Instance._from_object(source)
        instance.name = name
        return instance

    @property
    def _constructor_args(self):
        return set(inspect.getfullargspec(self.cls).args[1:])

    def _make_instance(self) -> object:
        """
        Create the real instance for this object
        """
        try:
            return object.__new__(self.cls)
        except TypeError as e:
            raise TypeError(
                f"Could not instantiate {self.name} of type {self.cls}"
            ) from e

    def __call__(self):
        """
        Create the real instance for this object, with child
        attributes attached.

        If the instance implements __setstate__ then this is
        called with a dictionary of instantiated children.
        """
        instance = self._make_instance()
        child_instances = {child.name: child() for child in self.children}

        if hasattr(instance, "__setstate__"):
            instance.__setstate__(child_instances)
        else:
            for name, child in child_instances.items():
                setattr(instance, name, child)

        from autofit import ModelObject

        if isinstance(instance, ModelObject) and (
            not hasattr(instance, "id") or instance.id is None
        ):
            instance.id = instance.next_id()

        return instance

    def _add_children(
        self, items: Union[ItemsView[str, Any], Iterable[Tuple[str, Any]]]
    ):
        """
        Add database representations of child attributes

        Parameters
        ----------
        items
            Attributes such as floats or priors that are associated
            with the real object
        """
        for key, value in items:
            if isinstance(value, property) or key.startswith("__") or key == "dtype":
                continue
            child = Object.from_object(value, name=key)
            self.children.append(child)

    class_path = sa.Column(sa.String)

    @property
    def cls(self) -> Type[object]:
        """
        The class of the real object
        """
        return get_class(self.class_path)

    @cls.setter
    def cls(self, cls: type):
        self.class_path = get_class_path(cls)
