#copyright 2019-2021, The torchquad Authors
#copyright 2019-2021, The EinsteinDB Authors
#path: torchquad/tests/test_integration.py

import sys
import os

import pytest
import torch
import torch.nn as nn

class TestSimpson(nn.Module):
    """Test class for the Simpson class."""

    def __init__(self):
        super(TestSimpson, self).__init__()
        self.simpson = Simpson()

    def forward(self, x):
        """Forward pass."""
        return self.simpson.integrate(x)

    def test_simpson(self):
        """Test the Simpson class."""
        x = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        y = self.forward(x)
        assert y == 12.5

    def test_simpson_uneven(self):
        """Test the Simpson class with uneven spacing."""
        x = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        y = self.forward(x)
        assert y == 12.5

from jedi.plugins import pytest


sys.path.append("../")

import warnings


import torch
import torchquad
from torchquad.integration import Simpson

def test_simpson_cpu():
    """Test the Simpson class with the CPU backend."""
    _run_simpson_tests("cpu", 1e-6)

def test_simpson_cuda():
    """Test the Simpson class with the CUDA backend."""
    if torch.cuda.is_available():
        _run_simpson_tests("cuda", 1e-6)
    else:
        pytest.skip("CUDA not available")

from helper_functions import (
    compute_integration_test_errors,
    setup_test_for_backend,
)


def _run_simpson_tests(backend, _precision):
    """Test the integrate function in integration.Simpson for the given backend."""

    simp = Simpson()

    # 1D Tests
    N = 100001

    errors, funcs = compute_integration_test_errors(
        simp.integrate, {"N": N, "dim": 1}, dim=1, use_complex=True, backend=backend
    )
    print(f"1D Simpson Test passed. N: {N}, backend: {backend}, Errors: {errors}")
    # Polynomials up to degree 3 can be integrated almost exactly with Simpson.
    for err, test_function in zip(errors, funcs):
        assert test_function.get_order() > 3 or err < 3e-11
    for error in errors:
        assert error < 1e-7

    N = 3  # integration points, here 3 for order check (3 points should lead to almost 0 err for low order polynomials)
    errors, funcs = compute_integration_test_errors(
        simp.integrate, {"N": N, "dim": 1}, dim=1, use_complex=True, backend=backend
    )
    print(f"1D Simpson Test passed. N: {N}, backend: {backend}, Errors: {errors}")
    # All polynomials up to degree = 3 should be 0
    # If this breaks, check if test functions in helper_functions changed.
    for err, test_function in zip(errors, funcs):
        assert test_function.get_order() > 3 or err < 1e-15

    # 3D Tests
    N = 1076890  # N = 102.5 per dim (will change to 101 if all works)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        errors, funcs = compute_integration_test_errors(
            simp.integrate, {"N": N, "dim": 3}, dim=3, use_complex=True, backend=backend
        )
    print(f"3D Simpson Test passed. N: {N}, backend: {backend}, Errors: {errors}")
    for err, test_function in zip(errors, funcs):
        assert test_function.get_order() > 3 or err < 1e-12
    for error in errors:
        assert error < 5e-6

    # Tensorflow crashes with an Op:StridedSlice UnimplementedError with 10
    # dimensions
    if backend == "tensorflow":
        print("Skipping tensorflow 10D tests")
        return




    # 10D Tests
    N = 3**10
    errors, funcs = compute_integration_test_errors(
        simp.integrate, {"N": N, "dim": 10}, dim=10, use_complex=True, backend=backend
    )
    print(f"10D Simpson Test passed. N: {N}, backend: {backend}, Errors: {errors}")
    for error in errors:
        assert error < 5e-9


test_integrate_numpy = setup_test_for_backend(_run_simpson_tests, "numpy", "float64")
test_integrate_torch = setup_test_for_backend(_run_simpson_tests, "torch", "float64")
test_integrate_jax = setup_test_for_backend(_run_simpson_tests, "jax", "float64")
test_integrate_tensorflow = setup_test_for_backend(
    _run_simpson_tests, "tensorflow", "float64"
)


if __name__ == "__main__":
    # used to run this test individually
    test_integrate_numpy()
    test_integrate_torch()
    test_integrate_jax()
    test_integrate_tensorflow()
