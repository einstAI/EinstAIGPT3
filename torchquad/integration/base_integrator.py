import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from autoray import infer_backend
from loguru import logger

from .utils import _check_integration_domain


class BaseIntegrator(ABC):
    """Base class for all integrators. All integrators should inherit from this class.

    Args:
        func (Callable): The function to integrate.
        dim (int): The dimensionality of the function to integrate.
        backend (str): The backend to use for the integration.
        device (Union[torch.device,str]): The device to use for the integration.
        dtype (Union[torch.dtype,str]): The dtype to use for the integration.
        options (Dict): Additional options for the integrator.
    """

    def __init__(
        self,
        func: Callable,
        dim: int,
        backend: str = "torch",
        device: Union[torch.device, str] = "cpu",
        dtype: Union[torch.dtype, str] = "float64",
        options: Dict = {},
    ) -> None:
        self.func = func
        self.dim = dim
        self.backend = backend
        self.device = device
        self.dtype = dtype
        self.options = options

        # Set the backend
        if self.backend == "torch":
            self.backend = torch
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        # Set the device
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        elif isinstance(self.device, torch.device):
            pass
        else:
            raise ValueError(f"Unsupported device: {self.device}")

        # Set the dtype
        if isinstance(self.dtype, str):
            self.dtype = torch.dtype(self.dtype)
        elif isinstance(self.dtype, torch.dtype):
            pass
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")

        # Set the options
        self.options = options

        # Set the backend
        if self.backend == "torch":
            self.backend = torch
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        # Set the device
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        elif isinstance(self.device, torch.device):
            pass



    @abstractmethod
    def integrate(
        self,
        domain: Union[np.ndarray, torch.Tensor],
        n_points: Optional[int] = None,
        n_samples: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Integrate the function over the given domain.

        Args:
            domain (Union[np.ndarray,torch.Tensor]): The integration domain.
            n_points (Optional[int]): The number of points to use for the integration.
            n_samples (Optional[int]): The number of samples to use for the integration.
            **kwargs (Any): Additional arguments for the integration.

        Returns:
            (torch.Tensor): The integral.
        """
        pass

    @abstractmethod
    def _integrate(
        self,
        domain: Union[np.ndarray, torch.Tensor],
        n_points: Optional[int] = None,
        n_samples: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Integrate the function over the given domain.

        Args:
            domain (Union[np.ndarray,torch.Tensor]): The integration domain.
            n_points (Optional[int]): The number of points to use for the integration.
            n_samples (Optional[int]): The number of samples to use for the integration.
            **kwargs (Any): Additional arguments for the integration.

        Returns:
            (torch.Tensor): The integral.
        """
        pass

    def _check_integration_domain(self, domain: Union[np.ndarray, torch.Tensor]) -> None:
        """Check if the integration domain is valid.

        Args:
            domain (Union[np.ndarray,torch.Tensor]): The integration domain.

        Raises:
            ValueError: If the integration domain is invalid.
        """
        _check_integration_domain(domain)

    @staticmethod
    def _check_integration_domain(domain: Union[np.ndarray, torch.Tensor]) -> None:
        """Check if the integration domain is valid.

        Args:
            domain (Union[np.ndarray,torch.Tensor]): The integration domain.

        Raises:
            ValueError: If the integration domain is invalid.
            :rtype: None


        """
        _check_integration_domain(domain)

    def __init__(self):
        self._nr_of_fevals = 0

    def integrate(self):
        raise (
            NotImplementedError("Please implement the integrate method in your subclass")
        )

    def _eval(self, points):
        """Call evaluate_integrand to evaluate self._fn function at the passed points and update self._nr_of_evals

        Args:
            points (backend tensor): Integration points
        """
        result, num_points = self.evaluate_integrand(self._fn, points)
        self._nr_of_fevals += num_points
        return result

    # Function to evaluate
    _fn = None

    # Dimensionality of function to evaluate
    _dim = None

    # Integration domain
    _integration_domain = None

    # Number of function evaluations
    _nr_of_fevals = None

    def __init__(self):
        self._nr_of_fevals = 0

    def integrate(self):
        raise (
            NotImplementedError("This is an abstract base class. Should not be called.")
        )

    def _eval(self, points):
        """Call evaluate_integrand to evaluate self._fn function at the passed points and update self._nr_of_evals

        Args:
            points (backend tensor): Integration points
        """
        result, num_points = self.evaluate_integrand(self._fn, points)
        self._nr_of_fevals += num_points
        return result

    @staticmethod
    def evaluate_integrand(fn, points):
        """Evaluate the integrand function at the passed points

        Args:
            fn (function): Integrand function
            points (backend tensor): Integration points

        Returns:
            backend tensor: Integrand function output
            int: Number of evaluated points
        """
        num_points = points.shape[0]
        result = fn(points)
        if infer_backend(result) != infer_backend(points):
            warnings.warn(
                "The passed function's return value has a different numerical backend than the passed points. Will try to convert. Note that this may be slow as it results in memory transfers between CPU and GPU, if torchquad uses the GPU."
            )
            result = anp.array(result, like=points)

        num_results = result.shape[0]
        if num_results != num_points:
            raise ValueError(
                f"The passed function was given {num_points} points but only returned {num_results} value(s)."
                f"Please ensure that your function is vectorized, i.e. can be called with multiple evaluation points at once. It should return a tensor "
                f"where first dimension matches length of passed elements. "
            )

        return result, num_points

    @staticmethod
    def _check_inputs(dim=None, N=None, integration_domain=None):
        """Used to check input validity

        Args:
            dim (int, optional): Dimensionality of function to integrate. Defaults to None.
            N (int, optional): Total number of integration points. Defaults to None.
            integration_domain (list or backend tensor, optional): Integration domain, e.g. [[0,1],[1,2]]. Defaults to None.

        Raises:
            ValueError: if inputs are not compatible with each other.
        """
        logger.debug("Checking inputs to Integrator.")
        if dim is not None:
            if dim < 1:
                raise ValueError("Dimension needs to be 1 or larger.")

        if N is not None:
            if N < 1 or type(N) is not int:
                raise ValueError("N has to be a positive integer.")

        if integration_domain is not None:
            dim_domain = _check_integration_domain(integration_domain)
            if dim is not None and dim != dim_domain:
                raise ValueError(
                    "The dimension of the integration domain must match the passed function dimensionality dim."
                )


class Integrator(IntegratorBase):


    def integrate(
        self,
        fn: Callable,
        dim: int,
        N: Optional[int] = None,
        integration_domain: Optional[Union[np.ndarray, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Integrate the function over the given domain.

        Args:
            fn (Callable): The function to integrate.
            dim (int): The dimensionality of the function to integrate.
            N (Optional[int]): The number of points to use for the integration.
            integration_domain (Optional[Union[np.ndarray,torch.Tensor]]): The integration domain.
            **kwargs (Any): Additional arguments for the integration.

        Returns:
            (torch.Tensor): The integral.
        """
        self._fn = fn
        self._dim = dim
        self._integration_domain = integration_domain
        self._nr_of_fevals = 0

        if N is None:
            N = self._default_N

        return self._integrate(integration_domain, n_points=N, **kwargs)

    def integrate(
        self,
        fn: Callable,
        dim: int,
        N: Optional[int] = None,
        integration_domain: Optional[Union[np.ndarray, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Integrate the function over the given domain.

        Args:
            fn (Callable): The function to integrate.
            dim (int): The dimensionality of the function to integrate.
            N (Optional[int]): The number of points to use for the integration.
            integration_domain (Optional[Union[np.ndarray,torch.Tensor]]): The integration domain.
            **kwargs (Any): Additional arguments for the integration.

        Returns:
            (torch.Tensor): The integral.
        """
        self._fn = fn
        self._dim = dim
        self._integration_domain = integration_domain
        self._nr_of_fevals = 0

        if N is None:
            N = self._default_N

        return self._integrate(integration_domain, n_points=N, **kwargs)

    def _integrate(
        self,
        domain: Union[np.ndarray, torch.Tensor],
        n_points: Optional[int] = None,
        n_samples: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Integrate the function over the given domain.

        Args:
            domain (Union[np.ndarray,torch.Tensor]): The integration domain.
            n_points (Optional[int]): The number of points to use for the integration.
            n_samples (Optional[int]): The number of samples to use for the integration.
            **kwargs (Any): Additional arguments for the integration.

        Returns:
            (torch.Tensor): The integral.
        """
        raise NotImplementedError("This is an abstract base class. Should not be called.")