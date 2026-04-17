"""Lazy JAX pytree registration for PyAutoFit's model and prior classes.

The classes themselves already carry the necessary ``tree_flatten`` /
``tree_unflatten`` methods. This module simply registers them with
``jax.tree_util`` on demand, via the lazy autoconf wrapper, so callers can
pass ``Model`` / ``Collection`` / ``ModelInstance`` / prior instances
through ``jax.jit`` and ``jax.vmap`` directly.
"""

from autoconf.jax_wrapper import register_pytree_node, register_pytree_node_class

_ENABLED = False
_REGISTERED_INSTANCE_CLASSES: set = set()
_CLASS_FIELD_CLASSIFIERS: dict = {}
_CLASS_CONSTRUCTOR_ARGS: dict = {}


def enable_pytrees() -> bool:
    """Register PyAutoFit model and prior classes as JAX pytree nodes.

    Returns ``True`` if registration was performed, ``False`` if JAX is not
    installed (in which case the call is a silent no-op). Safe to call more
    than once: subsequent calls return ``True`` without re-registering.
    """
    global _ENABLED
    if _ENABLED:
        return True

    try:
        import jax  # noqa: F401
    except ImportError:
        return False

    import autofit as af
    from autofit.mapper.prior_model.prior_model import Model
    from autofit.mapper.prior_model.collection import Collection

    for cls in (
        af.GaussianPrior,
        af.UniformPrior,
        af.LogGaussianPrior,
        af.LogUniformPrior,
        af.TruncatedGaussianPrior,
        Model,
        Collection,
        af.ModelInstance,
    ):
        try:
            register_pytree_node_class(cls)
        except ValueError:
            # Already registered (e.g. by another caller or a test fixture).
            # Re-registration is a JAX error but harmless for our purposes.
            pass

    _ENABLED = True
    return True


def register_model(model) -> bool:
    """Register every concrete ``model.cls`` in ``model`` as a JAX pytree node.

    PyAutoFit's ``Model.instance_flatten`` / ``instance_unflatten`` produce
    flatten/unflatten functions for instances of the user-defined class
    referenced by ``Model.cls`` (e.g. a ``Galaxy`` or ``Gaussian``). For JAX
    to recurse into those instances rather than treat them as opaque leaves,
    the user class itself must be registered with ``jax.tree_util``.

    This walks ``model`` (a ``Model`` or ``Collection``) and registers each
    ``cls`` it finds. Re-registering the same class is a silent no-op. Returns
    ``True`` if registration ran (or was already complete), ``False`` if JAX
    is missing.
    """
    if not enable_pytrees():
        return False

    from autofit.mapper.prior.abstract import Prior
    from autofit.mapper.prior_model.prior_model import Model
    from autofit.mapper.prior_model.collection import Collection
    from autofit.mapper.prior_model.abstract import AbstractPriorModel

    def _walk(node):
        if isinstance(node, Model):
            cls = node.cls
            classifier = _CLASS_FIELD_CLASSIFIERS.setdefault(cls, {})
            for name, value in node.items():
                is_dynamic = isinstance(value, (Prior, AbstractPriorModel))
                # setdefault: earliest classification wins. Different models
                # sharing the same cls (e.g. lens vs source Galaxy) may
                # declare different attribute sets; we accumulate them all.
                classifier.setdefault(name, is_dynamic)
            _CLASS_CONSTRUCTOR_ARGS.setdefault(
                cls, tuple(node.constructor_argument_names)
            )
            if cls not in _REGISTERED_INSTANCE_CLASSES:
                flatten, unflatten = _build_instance_pytree_funcs(cls)
                try:
                    register_pytree_node(cls, flatten, unflatten)
                except ValueError:
                    # Already registered elsewhere — keep going.
                    pass
                _REGISTERED_INSTANCE_CLASSES.add(cls)
            for _, sub in node.direct_prior_model_tuples:
                _walk(sub)
        elif isinstance(node, Collection):
            for _, sub in node.direct_prior_model_tuples:
                _walk(sub)

    _walk(model)
    return True


def _build_instance_pytree_funcs(cls):
    """Build flatten/unflatten functions for any instance of ``cls``.

    At flatten time we iterate the instance's own public attributes rather
    than a list captured from a single model at registration time. This is
    required because classes like ``Galaxy`` use ``**kwargs`` in ``__init__``,
    so different ``af.Model(Galaxy, ...)`` instances (e.g. a lens galaxy vs a
    source galaxy) produce instances with completely different attribute sets
    that nonetheless share the same ``cls`` — and JAX allows only one
    registration per class.

    Each attribute is classified as:

    * **Dynamic** (prior-derived): the corresponding ``Model`` attribute was
      a ``Prior`` or ``AbstractPriorModel``. Resolved to concrete numbers
      (or nested instances) per sampled point, so it becomes a JAX child leaf
      and gets traced under ``jax.jit``.
    * **Constant**: everything else — a fixed ``redshift=0.5``, or a concrete
      non-prior kwarg like ``Galaxy(pixelization=<Pixelization>)``. Goes into
      ``aux_data`` so it stays as the original Python object inside a trace.
      Required for control flow that reads constants
      (``sorted(..., key=lambda g: g.redshift)``) and for ``isinstance``
      dispatch on concrete kwargs (``isinstance(obj, Pixelization)``).

    Classification is read from the shared ``_CLASS_FIELD_CLASSIFIERS`` dict,
    which is updated by every ``register_model`` call. Attributes unknown to
    the classifier (never declared on any walked model) default to constant —
    safer than tracing an unknown object.
    """
    constructor_args = _CLASS_CONSTRUCTOR_ARGS.get(cls, ())
    constructor_arg_set = set(constructor_args)

    def _partition(instance):
        classifier = _CLASS_FIELD_CLASSIFIERS.get(cls, {})
        ctor_dyn: list = []
        ctor_const: list = []
        attr_dyn: list = []
        attr_const: list = []
        for name, value in vars(instance).items():
            if name.startswith("_") or name in ("cls", "id"):
                continue
            is_dynamic = classifier.get(name, False)
            in_ctor = name in constructor_arg_set
            if in_ctor and is_dynamic:
                ctor_dyn.append((name, value))
            elif in_ctor:
                ctor_const.append((name, value))
            elif is_dynamic:
                attr_dyn.append((name, value))
            else:
                attr_const.append((name, value))
        return ctor_dyn, ctor_const, attr_dyn, attr_const

    def flatten(instance):
        ctor_dyn, ctor_const, attr_dyn, attr_const = _partition(instance)
        children = (
            [v for _, v in ctor_dyn],
            [v for _, v in attr_dyn],
        )
        aux = (
            tuple(n for n, _ in ctor_dyn),
            tuple(n for n, _ in ctor_const),
            tuple(v for _, v in ctor_const),
            tuple(n for n, _ in attr_dyn),
            tuple(n for n, _ in attr_const),
            tuple(v for _, v in attr_const),
        )
        return children, aux

    def unflatten(aux, children):
        (
            dyn_names,
            const_names,
            const_vals,
            attr_dyn_names,
            attr_const_names,
            attr_const_vals,
        ) = aux
        dyn_vals, attr_dyn_vals = children
        kwargs = dict(zip(dyn_names, dyn_vals))
        kwargs.update(zip(const_names, const_vals))
        ordered = [kwargs[name] for name in constructor_args if name in kwargs]
        instance = cls(*ordered)
        for name, value in zip(attr_dyn_names, attr_dyn_vals):
            setattr(instance, name, value)
        for name, value in zip(attr_const_names, attr_const_vals):
            setattr(instance, name, value)
        return instance

    return flatten, unflatten
