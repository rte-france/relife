class LifetimeDataError(Exception):
    """Error raised when lifetime data arrays are incorrect

    Args:
        Exception (_type_): _description_
    """

    pass


class ArrayShapeError(Exception):
    """Error raised when array has incorrect shape

    Args:
        Exception (_type_): _description_
    """

    pass


class ArrayDimError(Exception):
    """Error raised when array has incorrect dimension

    Args:
        Exception (_type_): _description_
    """

    pass


class ArrayTypeError(Exception):
    """Error raised when array has incorrect type

    Args:
        Exception (_type_): _description_
    """

    pass


class FuncImplementationError(Exception):
    """Error raised when implementation of a model function is wrong

    Args:
        Exception (_type_): _description_
    """

    pass


class JacImplementationError(Exception):
    """Error raised when implementation of a jacobian method is wrong

    Args:
        Exception (_type_): _description_
    """

    pass


class InvalidFuncInput(Exception):
    """Error raised when func object passed to model is invalid

    Args:
        Exception (_type_): _description_
    """

    pass


class InvalidLikelihoodInput(Exception):
    """Error raised when likelihood object passed to model is invalid

    Args:
        Exception (_type_): _description_
    """

    pass


class InvalidOptimizerInput(Exception):
    """Error raised when optimizer object passed to model is invalid

    Args:
        Exception (_type_): _description_
    """

    pass
