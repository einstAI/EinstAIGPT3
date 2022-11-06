from typing import Callable, Union
from torchquad.integration import Simpson
from torchquad.utils import get_integrator




import torch
import torch.nn as nn
from torch import Tensor


class Trapezoid(Integrator):

    def __init__(self, N: int = 100, dim: int = 1, seed: int = 0):
            self.N = N
            self.dim = dim
            self.seed = seed
        
    def integrate(self, f: Callable, a: Union[Tensor, float], b: Union[Tensor, float]) -> Tensor:
        if self.dim == 1:
            return self._integrate_1d(f, a, b)
        elif self.dim == 2:
            return self._integrate_2d(f, a, b)
        else:
            raise NotImplementedError("Only 1D and 2D integration is supported")
        
    def _integrate_1d(self, f: Callable, a: Union[Tensor, float], b: Union[Tensor, float]) -> Tensor:
        if isinstance(a, float) and isinstance(b, float):
            x = torch.linspace(a, b, self.N, device=f.device, dtype=f.dtype)
            y = f(x)
            return (b - a) / (self.N - 1) * (y[0] + y[-1] + 2 * y[1:-1].sum())
        elif isinstance(a, Tensor) and isinstance(b, Tensor):
            x = torch.linspace(a, b, self.N, device=f.device, dtype=f.dtype)
            y = f(x)
            return (b - a) / (self.N - 1) * (y[..., 0] + y[..., -1] + 2 * y[..., 1:-1].sum(-1))
        else:
            raise NotImplementedError("Only float or Tensor integration is supported")

class Simpson:
    #     """Class for the Simpson integration method.
    #     Args:
    #         N (int): Number of integration points to use.
    #         dim (int): Dimension of the integration domain.
    #         seed (int): Seed for the random number generator.
    #     """
    #

def __init__(self, N: int = 100, dim: int = 1, seed: int = 0):
        self.N = N
        self.dim = dim
        self.seed = seed

    def integrate(
                    self,
        func: Callable,
        a: Union[float, Tensor],
        b: Union[float, Tensor],
        **kwargs,
    ):
    #         """Integrate a function of one or more variables.
        #         Args:
        #             func (Callable): The function to integrate. The function should take a tensor
        #                 as input and return a tensor of the same shape as the input.
        #             a (Tensor or float): Lower integration limit.
        #             b (Tensor or float): Upper integration limit.
        #             **kwargs: Additional arguments to pass to the function.
        #         Returns:
        #             Tensor: The integral of the function.
        #         """
        #

        if self.dim == 1:
            return self._integrate_1d(func, a, b, **kwargs)
        elif self.dim == 2:
            return self._integrate_2d(func, a, b, **kwargs)
        else:
            raise NotImplementedError(
                f"Integration of {self.dim} dimensions is not implemented."
            )

    def _integrate_1d(
        self,
        func: Callable,
        a: Union[float, Tensor],
        b: Union[float, Tensor],
        **kwargs,
    ):
        #         """Integrate a function of one variable.
        #         Args:
        #             func (Callable): The function to integrate. The function should take a tensor
        #                 as input and return a tensor of the same shape as the input.
        #             a (Tensor or float): Lower integration limit.
        #             b (Tensor or float): Upper integration limit.
        #             **kwargs: Additional arguments to pass to the function.
        #         Returns:
        #             Tensor: The integral of the function.
        #         """
        #

            if isinstance(a, float):
                a = torch.tensor(a, dtype=torch.float32)
            if isinstance(b, float):
                b = torch.tensor(b, dtype=torch.float32)

            x = torch.linspace(a, b, self.N, dtype=torch.float32)
            y = func(x, **kwargs)
            return (b - a) / (self.N - 1) * torch.sum(y[:-1] + y[1:])

        def _integrate_2d(
            self,
            func: Callable,
            a: Union[float, Tensor],
            b: Union[float, Tensor],
            **kwargs,
        ):
            #         """Integrate a function of two variables.
            #         Args:
            #             func (Callable): The function to integrate. The function should take a tensor
            #                 as input and return a tensor of the same shape as the input.
            #             a (Tensor or float): Lower integration limit.
            #             b (Tensor or float): Upper integration limit.
            #             **kwargs: Additional arguments to pass to the function.
            #         Returns:
            #             Tensor: The integral of the function.
            #         """
            #

                if isinstance(a, float):
                    a = torch.tensor(a, dtype=torch.float32)
                if isinstance(b, float):
                    b = torch.tensor(b, dtype=torch.float32)

                x = torch.linspace(a, b, self.N, dtype=torch.float32)
                y = torch.linspace(a, b, self.N, dtype=torch.float32)
                X, Y = torch.meshgrid(x, y)
                Z = func(torch.stack([X, Y], dim=-1), **kwargs)
                return (b - a) ** 2 / (self.N - 1) ** 2 * torch.sum(Z[:-1, :-1] + Z[1:, :-1] + Z[:-1, 1:] + Z[1:, 1:])

        def __repr__(self):
            return f"Simpson(N={self.N}, dim={self.dim}, seed={self.seed})"


class Simpson:
    """Class for the Simpson integration method.
    Args:
        N (int): Number of integration points to use.
        dim (int): Dimension of the integration domain.
        seed (int): Seed for the random number generator.
    """
    def __init__(self, N: int = 100, dim: int = 1, seed: int = 0):
        self.N = N
        self.dim = dim
        self.seed = seed

    def integrate(self, f: Callable, a: Union[Tensor, float], b: Union[Tensor, float]) -> Tensor:
        if self.dim == 1:
            return self._integrate_1d(f, a, b)
        elif self.dim == 2:
            return self._integrate_2d(f, a, b)
        else:
            raise NotImplementedError("Only 1D and 2D integration is supported")

    def _integrate_1d(self, f: Callable, a: Union[Tensor, float], b: Union[Tensor, float]) -> Tensor:
        if isinstance(a, float) and isinstance(b, float):
            x = torch.linspace(a, b, self.N, device=f.device, dtype=f.dtype)
            y = f(x)
            return (b - a) / (self.N - 1) * (y[0] + y[-1] + 2 * y[1:-1].sum())
        elif isinstance(a, Tensor) and isinstance(b, Tensor):
            x = torch.linspace(a, b, self.N, device=f.device, dtype=f.dtype)
            y = f(x)
            return (b - a) / (self.N - 1) * (y[..., 0] + y[..., -1] + 2 * y[..., 1:-1].sum(-1))
        else:
            raise NotImplementedError("Only float or Tensor integration is supported")

    def _integrate_2d(self, f: Callable, a: Union[Tensor, float], b: Union[Tensor, float]) -> Tensor:
        if isinstance(a, float) and isinstance(b, float):
            x = torch.linspace(a, b, self.N, device=f.device, dtype=f.dtype)
            y = torch.linspace(a, b, self.N, device=f.device, dtype=f.dtype)
            X, Y = torch.meshgrid(x, y)
            Z = f(torch.stack([X, Y], dim=-1))

            # Sum over the first and last row
            s = Z[0, :] + Z[-1, :]

            # Sum over the first and last column
            s += Z[:, 0] + Z[:, -1]

            # Sum over the interior points
            s += 4 * Z[1:-1, 1:-1].sum()

            # Sum over the points on the edges
            s += 2 * (Z[1:-1, 0] + Z[1:-1, -1] + Z[0, 1:-1] + Z[-1, 1:-1]).sum()

            return (b - a) ** 2 / (self.N - 1) ** 2 * s
        elif isinstance(a, Tensor) and isinstance(b, Tensor):
            x = torch.linspace(a, b, self.N, device=f.device, dtype=f.dtype)
            y = torch.linspace(a, b, self.N, device=f.device, dtype=f.dtype)
            X, Y = torch.meshgrid(x, y)
            Z = f(torch.stack([X, Y], dim=-1))

            # Sum over the first and last row
            s = Z[..., 0, :] + Z[..., -1, :]

            # Sum over the first and last column
            s += Z[..., :, 0] + Z[..., :, -1]

            # Sum over the interior points
            s += 4 * Z[..., 1:-1, 1:-1].sum(-1).sum(-1)

            # Sum over the points on the edges
            s += 2 * (Z[..., 1:-1, 0] + Z[..., 1:-1, -1] + Z[..., 0, 1:-1] + Z[..., -1, 1:-1]).sum(-1).sum(-1)

            return (b - a) ** 2 / (self.N - 1) ** 2 * s
        else:
            raise NotImplementedError("Only float or Tensor integration is supported")