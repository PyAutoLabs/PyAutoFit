from functools import wraps


class RecursionPromise:
    pass


def replace_promise(promise: RecursionPromise, obj, true_value, seen_objects=None):
    """
    Traverse the object replacing any identity of the promise with the true value

    Parameters
    ----------
    promise
        A placeholder for an object that had not been computed at the time some part of the object was computed
    obj
        An object computed that may contain Promises
    true_value
        The true value associated with the promise
    seen_objects
        A set of ids of objects that have already been checked in this promise replacement

    Returns
    -------
    obj
        The object with any identities of the Promise replaced
    """
    seen_objects = seen_objects or set()
    if id(obj) in seen_objects:
        return obj

    seen_objects.add(id(obj))

    if isinstance(obj, list):
        return [
            replace_promise(promise, item, true_value, seen_objects=seen_objects)
            for item in obj
        ]
    if isinstance(obj, dict):
        return {
            key: replace_promise(promise, value, true_value, seen_objects=seen_objects)
            for key, value in obj.items()
        }

    if obj is promise:
        return true_value
    try:
        for key, value in list(obj.__dict__.items()):
            setattr(
                obj,
                key,
                replace_promise(promise, value, true_value, seen_objects=seen_objects),
            )
    except (AttributeError, TypeError):
        pass
    return obj


class DynamicRecursionCache:
    def __init__(self):
        """
        A decorating class that prevents infinite loops when recursing graphs by attaching placeholders
        """
        self.cache = dict()

    def __call__(self, func):
        """
        Decorate the function to prevent recursion.

        When the function is called with a set of arguments, A, a Promise is stored for that set of arguments in the
        cache. If the function is called again with that set of arguments then the Promise is returned. When the
        function itself returns a value any identity of the Promise is replaced by the actual value returned.
        """

        @wraps(func)
        def wrapper(item, *args, **kwargs):
            item_id = id(item)

            if item_id in self.cache:
                return self.cache[item_id]
            recursion_promise = RecursionPromise()
            self.cache[item_id] = recursion_promise
            result = func(item, *args, **kwargs)
            result = replace_promise(recursion_promise, result, result)
            del self.cache[item_id]
            return result

        return wrapper
