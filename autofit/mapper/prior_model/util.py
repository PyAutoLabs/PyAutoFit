import inspect
from typing import Type, Dict
import typing

from autofit.mapper.prior_model.attribute_pair import AttributeNameValue


class PriorModelNameValue(AttributeNameValue):
    @property
    def prior_model(self):
        return self.value


def gather_namespaces(cls: Type) -> Dict[str, Dict]:
    """
    Recursively gather the globals and locals for a given class and its parent classes.
    """
    namespaces = {}

    try:
        for base in inspect.getmro(cls):
            if base is object:
                continue

            module = inspect.getmodule(base)
            if module:
                namespaces.update(vars(module))
    except AttributeError:
        pass

    namespaces.update(vars(typing))

    return namespaces
