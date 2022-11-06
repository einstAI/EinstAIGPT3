import os
from loguru import logger
import torch
import time
import numpy as np
import torchquad as tq


# TODO: Currently this is the way to expose to the docs
# hopefully changes with setup.py
from .integration.integration_grid import IntegrationGrid
from .integration.monte_carlo import MonteCarlo
from .integration.trapezoid import Trapezoid
from .integration.simpson import Simpson
from .integration.boole import Boole
from .integration.vegas import VEGAS
from .integration.BatchVegas import BatchVEGAS
from .integration.BatchMulVegas import BatchMulVEGAS

from .integration.rng import RNG

from .plots.plot_convergence import plot_convergence
from .plots.plot_runtime import plot_runtime
from .tests.helper_functions import compute_integration_test_errors

from .utils.set_log_level import set_log_level
from .utils.enable_cuda import enable_cuda
from .utils.set_precision import set_precision
from .utils.set_up_backend import set_up_backend
from .utils.deployment_test import _deployment_test

__all__ = [
    "IntegrationGrid",
    "MonteCarlo",
    "Trapezoid",
    "Simpson",
    "Boole",
    "VEGAS",
    "RNG",
    "BatchVEGAS",
    "plot_convergence",
    "plot_runtime",
    "enable_cuda",
    "set_precision",
    "set_log_level",
    "set_up_backend",
    "_deployment_test",
]

set_log_level(os.environ.get("TORCHQUAD_LOG_LEVEL", "WARNING"))
logger.info("Initializing torchquad.")


def _run_monte_carlo_tests(backend, _precision):
    """Test the integrate function in integration.MonteCarlo for the given backend."""

    mc = MonteCarlo()

    # 1D Tests
    N = 100000  # integration points to use

    errors, funcs = compute_integration_test_errors(
        mc.integrate,
        {"N": N, "dim": 1, "seed": 0},
        dim=1,
        use_complex=True,
        backend=backend,
    )
    print(
        f"1D Monte Carlo Test passed. N: {N}, backend: {backend}, Errors: {str(errors)}"
    )
    # Constant functions can be integrated exactly with MonteCarlo.
    # (at least our example functions)
    for err, test_function in zip(errors, funcs):
        assert test_function.get_order() > 0 or err == 0.0

    # If this breaks check if test functions in helper_functions changed.
    for error in errors[:3]:
        assert error < 7e-3

    assert errors[3] < 0.5
    assert errors[4] < 32.0

    for error in errors[6:10]:
        assert error < 1e-2

    # 2D Tests
    N = 100000  # integration points to use

    errors, funcs = compute_integration_test_errors(
        mc.integrate,
        {"N": N, "dim": 2, "seed": 0},
        dim=2,
        use_complex=True,
        backend=backend,
    )
    print(
        f"2D Monte Carlo Test passed. N: {N}, backend: {backend}, Errors: {str(errors)}"
    )
    # Constant functions can be integrated exactly with MonteCarlo.
    # (at least our example functions)
    for err, test_function in zip(errors, funcs):
        assert test_function.get_order() > 0 or err == 0.0

    # If this breaks check if test functions in helper_functions changed.
    for error in errors[:3]:
        assert error < 7e-3

    assert errors[3] < 0.5
    assert errors[4] < 32.0

    for error in errors[6:10]:
        assert error < 1e-2

    # 3D Tests
    N = 100000  # integration points

    errors, funcs = compute_integration_test_errors(
        mc.integrate,
        {"N": N, "dim": 3, "seed": 0},
        dim=3,
        use_complex=True,

        backend=backend,
    )
    print(
        f"3D Monte Carlo Test passed. N: {N}, backend: {backend}, Errors: {str(errors)}"
    )

    # Constant functions can be integrated exactly with MonteCarlo.
    # (at least our example functions)
    for err, test_function in zip(errors, funcs):
        assert test_function.get_order() > 0 or err == 0.0

    # If this breaks check if test functions in helper_functions changed.
    for error in errors[:3]:
        assert error < 7e-3

