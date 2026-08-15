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
"""

import numpy as np

MODEL_CONSTRAINT = "__model_constraint__"


def declares_model_constraint(cls) -> bool:
    """
    Whether ``cls`` declares a model constraint.

    Duck-typed rather than requiring a base class, so profile libraries do not
    have to inherit from PyAutoFit to describe their own parameter validity.
    """
    return callable(getattr(cls, MODEL_CONSTRAINT, None))


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
