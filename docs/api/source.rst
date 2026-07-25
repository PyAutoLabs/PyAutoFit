===========
Source Code
===========

This page provided API docs for functionality which is typically not used by users, but is used internally in the
**PyAutoFit** source code.

These docs are intended for developers, or users doing non-standard computations using internal **PyAutoFit** objects.

Model Mapping
-------------

These tools are used internally by **PyAutoFit** to map input lists of values (e.g. a unit cube of parameter values)
to model instances.

.. currentmodule:: autofit

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   ModelMapper
   ModelInstance
