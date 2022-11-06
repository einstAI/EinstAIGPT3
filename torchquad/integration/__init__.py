from typing import Callable, Union

import torch
from torch import Tensor


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



