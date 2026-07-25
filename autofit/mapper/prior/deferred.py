from autofit import exc


class DeferredInstance:
    def __init__(self, cls: type, constructor_arguments: {str: object}):
        """
        An instance that has been deferred for later construction

        Parameters
        ----------
        cls
            The class to be constructed
        constructor_arguments
            The arguments provided by the optimiser
        """
        self.cls = cls
        self.constructor_arguments = constructor_arguments

    @property
    def deferred_argument_names(self) -> [str]:
        """
        The names of arguments still required to instantiate the class
        """
        return [
            name
            for name, value in self.constructor_arguments.items()
            if isinstance(value, DeferredArgument)
        ]

    def __call__(self, **kwargs):
        """
        Constructs an instance of the class provided that all unset arguments are
        passed.

        Parameters
        ----------
        kwargs
            Key value pairs for arguments that should be set

        Returns
        -------
        instance: self.cls
            An instance of the class
        """
        return self.cls(**{**self.constructor_arguments, **kwargs})

    def __getattr__(self, item):
        """
        Failing to get an attribute is considered to indicate an attempt to use a
        deferred instance without first instantiating it. As such an exception is
        raised to warn the user that they need to instantiate the class.
        """
        try:
            super().__getattribute__(item)
        except AttributeError:
            raise exc.DeferredInstanceException(
                f"{self.cls.__name__} cannot be called until it is instantiated with"
                f" deferred arguments {self.deferred_argument_names}"
            )


class DeferredArgument:
    """
    A deferred argument which is passed into the construct the final instance after
    model mapper instance generation
    """

    pass
