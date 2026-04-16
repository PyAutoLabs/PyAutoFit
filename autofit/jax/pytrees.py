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

    from autofit.mapper.prior_model.prior_model import Model
    from autofit.mapper.prior_model.collection import Collection

    def _walk(node):
        if isinstance(node, Model):
            cls = node.cls
            if cls not in _REGISTERED_INSTANCE_CLASSES:
                flatten, unflatten = _build_instance_pytree_funcs(node)
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


def _build_instance_pytree_funcs(model):
    """Build flatten/unflatten functions for instances of ``model.cls``.

    Constants from the original model definition (e.g. ``Galaxy(redshift=0.5)``)
    are placed in the JAX ``aux_data`` so they remain concrete Python values
    inside a ``jax.jit`` trace. Only prior-derived constructor arguments are
    placed in ``children`` (and therefore become JAX tracers).

    This is critical for code that uses constants for control flow — e.g.
    ``sorted(galaxies, key=lambda g: g.redshift)`` in ``Tracer`` — which would
    otherwise raise ``TracerBoolConversionError`` under JIT.
    """
    constructor_args = list(model.constructor_argument_names)
    constant_arg_names = [
        name for name in constructor_args if name in dict(model.direct_instance_tuples)
    ]
    constant_values = {
        name: dict(model.direct_instance_tuples)[name] for name in constant_arg_names
    }
    dynamic_arg_names = [
        name for name in constructor_args if name not in constant_arg_names
    ]

    def flatten(instance):
        attribute_names = [
            name
            for name in model.direct_argument_names
            if hasattr(instance, name) and name not in constructor_args
        ]
        children = (
            [getattr(instance, name) for name in dynamic_arg_names],
            [getattr(instance, name) for name in attribute_names],
        )
        aux = (
            tuple(dynamic_arg_names),
            tuple(constant_arg_names),
            tuple(constant_values[n] for n in constant_arg_names),
            tuple(attribute_names),
        )
        return children, aux

    def unflatten(aux, children):
        dyn_names, const_names, const_vals, attr_names = aux
        dyn_vals, attr_vals = children
        kwargs = dict(zip(dyn_names, dyn_vals))
        kwargs.update(zip(const_names, const_vals))
        ordered = [kwargs[name] for name in constructor_args]
        instance = model.cls(*ordered)
        for name, value in zip(attr_names, attr_vals):
            setattr(instance, name, value)
        return instance

    return flatten, unflatten
