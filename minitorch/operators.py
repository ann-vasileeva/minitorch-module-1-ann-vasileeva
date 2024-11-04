"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def mul(x: float, y: float) -> float:
    """Multiplies two numbers
    Args:
        x: first number
        y: second number
    Returns:
        x * y, multipication result

    """
    return float(x * y)


def id(x: float) -> float:
    """Returns the input unchanged
    Args:
        x: input
    Returns:
        x unchanged input
    """
    return float(x)


def add(x: float, y: float) -> float:
    """Adds two numbers
    Args:
        x: first number
        y: second number
    Returns:
        x + y, sum result
    """
    return float(x + y)


def neg(x: float) -> float:
    """Negates a number
    Args:
        x: input
    Returns:
        x negated input
    """
    return -float(x)


def lt(x: float, y: float) -> bool:
    """Checks if one number is less than another
    Args:
        x: first number
        y: second number
    Returns:
        x < y, if x is less than y
    """
    return float(x < y)


def eq(x: float, y: float) -> bool:
    """Checks if two numbers are equal
    Args:
        x: first number
        y: second number
    Returns:
        x == y, if x is equal to y
    """
    return float(x == y)


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers
    Args:
        x: first number
        y: second number
    Returns:
        max(x,y), the larger of two numbers
    """
    if lt(x, y):
        return y
    return x


def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are close in value
    Args:
        x: first number
        y: second number
    Returns:
        $f(x) = |x - y| < 1e-2$
    """
    return math.fabs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function
    # $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
    Args:
        x: input
    Returns:
        sigmoid(x)
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Applies the ReLU activation function
    Args:
        x: input
    Returns:
        relu(x) = max(0, x)
    """
    return float(max(0, x))


def log(x: float) -> float:
    """Calculates the natural logarithm
    Args:
        x: input
    Returns:
        ln(x)
    """
    return math.log(x)


def exp(x: float) -> float:
    """Calculates the exponential function
    Args:
        x: input
    Returns:
        exp(x)
    """
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the reciprocal
    Args:
        x: input
    Returns:
        1 / x
    """
    return 1 / x


def log_back(x: float, y: float) -> float:
    """Computes the derivative of log times a second arg
    Args:
        x: input
        y: second arg
    Returns:
        (1 / x) * y
    """
    return y / x


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of reciprocal times a second arg
    Args:
        x: input
        y: second arg
    Returns:
         - y * (1/x**2)
    """
    return -y / (x**2)


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of ReLU times a second arg
    Args:
        x: input
        y: second arg
    Returns:
        grad_output * [x > 0]
    """
    if x < 0:
        return 0
    return y


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
# docstrings were generated with chat gpt


def map(func: Callable, lst: Iterable[float]) -> Iterable[float]:
    """Higher-order function that applies a given function to each element of an iterable.

    Args:
    ----
        func (function): A function to apply to each element.
        lst (iterable): The iterable to map over.

    Returns:
    -------
        list: A new list containing the results of applying func to each element of lst.

    """
    return [func(elem) for elem in lst]


def zipWith(
    func: Callable, lst1: Iterable[float], lst2: Iterable[float]
) -> Iterable[float]:
    """Higher-order function that combines elements from two iterables using a given function.

    Args:
    ----
        func (function): A function to combine elements from lst1 and lst2.
        lst1 (iterable): The first iterable.
        lst2 (iterable): The second iterable.

    Returns:
    -------
        list: A new list consisting of the results of applying func to pairs of elements from lst1 and lst2.

    """
    return [func(lst1[i], lst2[i]) for i in range(len(lst1))]


def reduce(func: Callable, lst: Iterable[float]) -> float:
    """Higher-order function that reduces an iterable to a single value using a given function.

    Args:
    ----
        func (function): A function to apply.
        lst (iterable): The iterable to reduce.

    Returns:
    -------
        reduced: The reduced value.

    """
    if len(lst) == 0:
        return 0
    reduced = lst[0]
    for elem in lst[1:]:
        reduced = func(reduced, elem)
    return reduced


def negList(lst: Iterable) -> Iterable:
    """Negate all elements in a list using map.

    Args:
    ----
        lst (list): The list of numbers to negate.

    Returns:
    -------
        list: A new list with the negated values.

    """
    return map(neg, lst)


def addLists(lst1: Iterable, lst2: Iterable) -> Iterable:
    """Add corresponding elements from two lists using zipWith.

    Args:
    ----
        lst1 (list): The first list.
        lst2 (list): The second list.

    Returns:
    -------
        list: A new list with the sums of corresponding elements from lst1 and lst2.

    """
    return zipWith(add, lst1, lst2)


def sum(lst: Iterable) -> float:
    """Sum all elements in a list using reduce.

    Args:
    ----
        lst (list): The list of numbers to sum.

    Returns:
    -------
        float: The total sum of the list.

    """
    return reduce(add, lst)


def prod(lst: Iterable) -> float:
    """Calculate the product of all elements in a list using reduce.

    Args:
    ----
        lst (list): The list of numbers to multiply.

    Returns:
    -------
        float: The product of the list.

    """
    return reduce(mul, lst)
