"""
torecsys.utils.logging.decorator is a sub model of utils including decorator functions
to tag features of functions.
"""
import warnings
from functools import wraps


def in_development(func: callable):
    """a decorator to write a message in a layer or a estimator where they have not been tested
    
    Args:
        func (callable): a callable function, and most likely they are torecsys.layers or torecsys.estimators
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn("This function is in developing.", UserWarning)
        return func(*args, **kwargs)

    return wrapper


def no_jit_experimental(func: callable):
    """a decorator to write a message in a layer or a estimator where they have been checked
    to be non-compatible with torch.jit.trace
    
    Args:
        func (callable): a callable function, and most likely they are torecsys.layers or torecsys.estimators
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn("This function is not allowed to used in jit-mode.", UserWarning)
        return func(*args, **kwargs)

    return wrapper


def no_jit_experimental_by_named_tensor(func: callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            "The model is checked that it is not compatible with torch.jit.trace " +
            "due to the NamedTensor method. This will be updated to compatibility " +
            "when PyTorch update.", UserWarning
        )
        return func(*args, **kwargs)

    return wrapper


def jit_experimental(func: callable):
    """a decorator to write a message in a layer or a estimator where they have been checked
    to be compatible with torch.jit.trace
    
    Args:
        func (callable): a callable function, and most likely they are torecsys.layers or torecsys.estimators
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn("This function is allowed to used in jit-mode, but the feature is in experimental." +
                      "Please do not use it for anything important until they are released as stable.", UserWarning)
        return func(*args, **kwargs)

    return wrapper
