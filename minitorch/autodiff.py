from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Set

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    new_vals_plus = list(vals)
    new_vals_minus = list(vals)
    
    new_vals_plus[arg] += epsilon
    new_vals_minus[arg] -= epsilon
    
    f_plus = f(*new_vals_plus)
    f_minus = f(*new_vals_minus)
    return (f_plus - f_minus) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    top_order = [variable]
    visited = set()
    
    def visit(variable: Variable) -> None:
        if variable.unique_id in visited:
            return 
        visited.add(variable.unique_id)
        if variable.history is not None:
            for neigh in variable.history.inputs:
                if neigh.unique_id not in visited:
                    visit(neigh)
            top_order.append(variable)
    
    visit(variable)
    return top_order[::-1]     


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    top_order = topological_sort(variable)
    cur_derivs = {variable.unique_id: deriv}
    
    for node in top_order:
        if node.unique_id not in cur_derivs:
            cur_derivs[node.unique_id] = 0
        cur_der = cur_derivs[node.unique_id]
        if node.is_leaf():
            node.accumulate_derivative(cur_der)
        else:
            chain_seq = node.chain_rule(cur_der)
            for inp, grad in chain_seq:
                if inp.unique_id in cur_derivs:
                    cur_derivs[inp.unique_id] += grad
                else:
                    cur_derivs[inp.unique_id] = grad
                
            
@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
