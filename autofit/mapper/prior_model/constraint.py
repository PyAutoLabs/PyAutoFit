"""
Class-declared model constraints, evaluated as traced values.

This is the differentiable sibling of :meth:`AbstractPriorModel.add_assertion`.
Assertions are attached to a *model instance* by the user and signal failure by
raising :class:`FitException`; that works on the NumPy path, where
:class:`Fitness` catches the exception and returns the resample sentinel, but it
cannot work under JAX. A ``raise`` needs a concrete boolean, and inside a trace
the condition is a tracer — attempting it gives ``TracerBoolConversionError``.
That is why guards such as ``autogalaxy.profiles.validate.validate_ell_comps``
return early for non-concrete scalars rather than raising: the escape hatch is
load-bearing, or every jitted likelihood would crash instead of sampling.

A model constraint differs in exactly two ways:

- it is declared **on the class**, so every model built from that class carries
  it without the user remembering to attach anything;
- it is evaluated as a **traced non-negative violation measure** rather than
  raising, so it survives ``jit``, ``vmap`` and ``grad``.

Declaring one
-------------

A class declares a constraint by defining ``__model_constraint__``, which takes
the array module and returns a non-negative violation measure — ``0`` when the
constraint is satisfied, growing with the severity of the violation:

.. code-block:: python

    class EllProfile:
        def __model_constraint__(self, xp=np):
            magnitude = xp.sqrt(
                self.ell_comps[0] ** 2 + self.ell_comps[1] ** 2
            )
            return xp.maximum(magnitude - 0.999, 0.0)

A *measure* rather than a boolean deliberately: a bool is enough for the
diagnostic counters that consume this today, but a magnitude is what a penalty
term would need later, and it carries a usable gradient back into the valid
region. Returning a bool would work for counting and then have to be redesigned.

Declaring a ball
----------------

``__model_constraint__`` *measures* a violation; it cannot say how to fix one.
A search that wants to keep its parameters inside the valid region needs the
constraint's **structure**, not just its magnitude, and for the commonest case —
a pair of parameters confined to a disk, ``e0**2 + e1**2 < r**2`` — the structure
is fully described by the coordinates involved and the radius.

``__model_ball_constraints__`` declares exactly that, as a class attribute (not a
method: it is static geometry, resolved once from the model rather than evaluated
per step):

.. code-block:: python

    class EllProfile:
        __model_ball_constraints__ = ((("ell_comps",), 0.999),)

Each entry is a ``(path, radius)`` pair. ``path`` is a tuple of attribute names
navigating from the declaring component to the
:class:`~autofit.mapper.prior.tuple_prior.TuplePrior` whose components are the
ball's coordinates; ``radius`` is the radius the coordinates are confined within.

This is deliberately **not** a second statement of the same fact as
``__model_constraint__``, but the projectable form of it. The measure is what a
counter or a penalty term reads; the ball is what
:class:`~autofit.non_linear.clipper.ClipperPriorBoxJoint` projects onto. A class
may declare either, or both, independently.

The radius a class declares should be the threshold **its own maths** needs, not
the boundary of formal validity. For ``ell_comps`` those differ: the geometry is
undefined at magnitude ``1``, but the conversion to an axis ratio saturates at
``0.999``, and the annulus between them is a dead-gradient region a search can
sit in while still passing every validity check. Projecting onto the clamp keeps
lanes out of it; projecting onto ``1 - epsilon`` would park them in it.
"""

import numpy as np

MODEL_CONSTRAINT = "__model_constraint__"
MODEL_BALL_CONSTRAINT = "__model_ball_constraints__"


def declares_model_constraint(cls) -> bool:
    """
    Whether ``cls`` declares a model constraint.

    Duck-typed rather than requiring a base class, so profile libraries do not
    have to inherit from PyAutoFit to describe their own parameter validity.
    """
    return callable(getattr(cls, MODEL_CONSTRAINT, None))


def declares_ball_constraints(cls) -> bool:
    """
    Whether ``cls`` declares one or more ball constraints.

    Duck-typed on the presence of a non-empty ``__model_ball_constraints__``,
    exactly like :func:`declares_model_constraint`, so a profile library
    describes its own geometry without inheriting from PyAutoFit.
    """
    return bool(getattr(cls, MODEL_BALL_CONSTRAINT, None))


def ball_constraints_for(cls) -> tuple:
    """
    The ``((path, radius), ...)`` ball constraints ``cls`` declares, normalised.

    Each ``path`` is returned as a tuple of attribute names and each ``radius``
    as a float, so callers never have to defend against a class having written
    a list where a tuple was expected.

    Parameters
    ----------
    cls
        A class, which may or may not declare ``__model_ball_constraints__``.

    Returns
    -------
    An empty tuple when nothing is declared.
    """
    declared = getattr(cls, MODEL_BALL_CONSTRAINT, None)
    if not declared:
        return ()

    normalised = []
    for entry in declared:
        try:
            path, radius = entry
        except (TypeError, ValueError) as e:
            raise AssertionError(
                f"{cls.__name__}.{MODEL_BALL_CONSTRAINT} entries must be "
                f"(path, radius) pairs; got {entry!r}"
            ) from e

        if isinstance(path, str):
            raise AssertionError(
                f"{cls.__name__}.{MODEL_BALL_CONSTRAINT} paths must be a tuple "
                f"of attribute names, not a bare string; got {path!r}"
            )

        normalised.append((tuple(path), float(radius)))

    return tuple(normalised)


def violation_for_instance(instance, xp=np):
    """
    The non-negative violation measure ``instance`` reports for itself.

    Parameters
    ----------
    instance
        An instance of a class declaring ``__model_constraint__``.
    xp
        The array module, ``numpy`` or ``jax.numpy``.
    """
    return getattr(instance, MODEL_CONSTRAINT)(xp=xp)
