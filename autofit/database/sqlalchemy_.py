"""
All usage of SQLAlchemy should be imported via this module to allow SQLAlchemy to be
installed optionally.

Sufficient interface is implemented to permit import of SQLAlchemy based classes without
any error. If any attempt is made to use those classes a meaningful warning is returned.

``sa`` is a lazy proxy: the real ``sqlalchemy`` module is imported on first
attribute access, not when this module is imported, so sessions that never use
the database do not pay sqlalchemy's import cost. Modules that reference
``sa.<attr>`` in function signatures must use ``from __future__ import
annotations`` so the annotation does not trigger the import at definition time.
"""


def fail():
    raise ImportError(
        "Please install SQLAlchemy to use the database"
    )


class MockBase:
    def __init__(self, *args, **kwargs):
        fail()


class MockSQlAlchemy:
    def __getattr__(self, item):
        return self

    def __call__(self, *args, **kwargs):
        return self

    def __mro_entries__(self, *args, **kwargs):
        return tuple()

    def declarative_base(self):
        return MockBase

    def __getitem__(self, item):
        fail()

    def __setitem__(self, key, value):
        fail()


MockBase.metadata = MockSQlAlchemy()


class _LazySQLAlchemy:
    """
    Defers ``import sqlalchemy`` until an attribute is first accessed, then
    delegates every attribute to the real module (or to ``MockSQlAlchemy``
    when sqlalchemy is not installed).
    """

    _target = None

    def _load(self):
        if _LazySQLAlchemy._target is None:
            try:
                import sqlalchemy

                _LazySQLAlchemy._target = sqlalchemy
            except ImportError:
                _LazySQLAlchemy._target = MockSQlAlchemy()
        return _LazySQLAlchemy._target

    def __getattr__(self, item):
        return getattr(self._load(), item)

    def __call__(self, *args, **kwargs):
        return self._load()(*args, **kwargs)

    def __mro_entries__(self, *args, **kwargs):
        target = self._load()
        if isinstance(target, MockSQlAlchemy):
            return tuple()
        raise TypeError(f"{target!r} cannot be used as a base class")

    def __getitem__(self, item):
        return self._load()[item]

    def __setitem__(self, key, value):
        self._load()[key] = value


class _LazyDeclarative(_LazySQLAlchemy):
    def _load(self):
        if _LazyDeclarative._target is None:
            try:
                from sqlalchemy.ext import declarative

                _LazyDeclarative._target = declarative
            except ImportError:
                _LazyDeclarative._target = MockSQlAlchemy()
        return _LazyDeclarative._target


_LazyDeclarative._target = None

sa = _LazySQLAlchemy()
declarative = _LazyDeclarative()
