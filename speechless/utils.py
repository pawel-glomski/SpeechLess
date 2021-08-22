import logging
import numpy as np
from typing import List, Tuple

Real = np.double

NULL_LOGGER = logging.getLogger('null')
NULL_LOGGER.handlers = [logging.NullHandler()]
NULL_LOGGER.propagate = False


def rangesOfEquality(arr: np.ndarray) -> List[Tuple[int, int]]:
    """Returns a list of ranges, for which elements of a specified array are equal. For example:
    an array [1,1,2,2,3,3] will generate the output: [(0, 2),(2,4),(4,6)].

    Args:
        arr (np.ndarray): 1-dimensional array to check

    Returns:
        List[Tuple[int, int]]: A list of ranges: <start, end)
    """
    assert len(arr.shape) == 1

    ranges = np.where(arr[:-1] != arr[1:])[0] + 1
    isEven = len(ranges) % 2 == 0
    if (isEven and arr[0]) or (not isEven and arr[0]):
        ranges = np.concatenate([[0], ranges])
    if (isEven and arr[0]) or (not isEven and not arr[0]):
        ranges = np.concatenate([ranges, [len(arr)]])
    return ranges.reshape((-1, 2))


def intLinspaceStepsByLimit(start: int, stop: int, partLimit: int) -> np.ndarray:
    """Splits an interval into parts of roughly equal sizes, where the maximum size of a part is 
    limited, and returns sizes of these parts.

    Args:
        start (int): Start of an interval
        stop (int): Endpoint of an interval
        partLimit (int): Size limit of a part

    Returns:
        np.ndarray: List of values
    """
    return intLinspaceStepsByNo(start, stop, intNumberOfParts(stop - start, partLimit))


def intLinspaceStepsByNo(start: int, stop: int, num: int) -> np.ndarray:
    """Splits an interval into a specified number of parts of roughly equal sizes and returns sizes 
    of these parts.

    Args:
        start (int): Start of an interval
        stop (int): Endpoint of an interval
        num (int): Number of parts

    Returns:
        np.ndarray: List of values
    """
    vals = np.linspace(start, stop, num + 1, endpoint=True, dtype=int)
    return vals if len(vals) == 1 else (vals[1:] - vals[:-1])


def intLinspaceStepsByLimit(start: int, stop: int, partLimit: int) -> np.ndarray:
    """Splits an interval into parts of roughly equal sizes, where the maximum size of a part is 
    limited, and returns sizes of these parts.

    Args:
        start (int): Start of an interval
        stop (int): Endpoint of an interval
        partLimit (int): Size limit of a part

    Returns:
        np.ndarray: List of values
    """
    return intLinspaceStepsByNo(start, stop, intNumberOfParts(stop - start, partLimit))


def intNumberOfParts(number: int, partLimit: int) -> int:
    """Calculates the number of parts needed to evenly divide a number when the maximum value of a 
    part is limited

    Args:
        number (int): Number to divide
        partLimit (int): Limit of a part

    Returns:
        int: Number of parts
    """
    return int(np.ceil(number / partLimit))
