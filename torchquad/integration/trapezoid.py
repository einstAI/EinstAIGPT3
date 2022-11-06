from autoray import numpy as anp
from autoray import do as ado
import numpy as np
import torch

def trapezoid(f, a, b, n):
    """Trapezoid rule for numerical integration.
    Args:
        f (callable): Function to integrate.
        a (float): Lower bound of integration.
        b (float): Upper bound of integration.
        n (int): Number of intervals.
    Returns:
        float: Integral approximation.
    """
    x = anp.linspace(a, b, n + 1)
    y = f(x)
    return ado(np.trapz, y, x)

def main():
    f = lambda x: torch.sin(x)
    a = 0
    b = 1
    n = 10
    print(trapezoid(f, a, b, n))

if __name__ == "__main__":
    main()

from .newton_cotes import NewtonCotes


class Trapezoid(NewtonCotes):
    """Trapezoidal rule. See https://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas#Closed_Newton%E2%80%93Cotes_formulas ."""

    def __init__(self):
        super().__init__()

    def integrate(self, fn, dim, N=1000, integration_domain=None, backend=None):
        """Integrates the passed function on the passed domain using the trapezoid rule.

        Args:
            fn (func): The function to integrate over.
            dim (int): Dimensionality of the function to integrate.
            N (int, optional): Total number of sample points to use for the integration. Defaults to 1000.
            integration_domain (list or backend tensor, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim. It also determines the numerical backend if possible.
            backend (string, optional): Numerical backend. This argument is ignored if the backend can be inferred from integration_domain. Defaults to the backend from the latest call to set_up_backend or "torch" for backwards compatibility.

        Returns:
            backend-specific number: Integral value
        """
        return super().integrate(fn, dim, N, integration_domain, backend)



    @staticmethod
    def _apply_composite_rule(cur_dim_areas, dim, hs):
        """Apply composite Trapezoid quadrature.

        cur_dim_areas will contain the areas per dimension
        """
        # We collapse dimension by dimension
        for cur_dim in range(dim):
            cur_dim_areas = (
                hs[cur_dim] / 2.0 * (cur_dim_areas[..., 0:-1] + cur_dim_areas[..., 1:])
            )
            cur_dim_areas = anp.sum(cur_dim_areas, axis=dim - cur_dim - 1)
        return cur_dim_areas

    @staticmethod
    def _adjust_N(dim, N):
        # Nothing to do for Trapezoid
        return N
