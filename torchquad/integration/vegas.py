import self
from autoray import numpy as anp
from autoray import infer_backend, astype
from functools import partial
from loguru import logger
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.utils.data.dataloader as dataloader
import torch.utils.data.dataset as dataset
import torch.utils.data.sampler as sampler
import torchvision


from .base_integrator import BaseIntegrator
from .utils import _setup_integration_domain
from .rng import RNG
from .vegas_map import VEGASMap
from .vegas_stratification import VEGASStratification




class VEGAS(BaseIntegrator):
    """VEGAS is a Monte Carlo integration algorithm that uses stratification and importance sampling.

    Args:
        integration_domain (list or backend tensor): Integration domain, e.g. [[-1,1],[0,1]].
        n_dim (int): Dimension of the integration domain. If this is specified, the integration_domain is ignored.
        n_eval (int): Number of evaluations.
        batch_size (int): Number of function evaluations per batch.
        n_stratification (int): Number of stratification samples.
        n_refinements (int): Number of refinement iterations.
        n_importance (int): Number of importance samples.
        stratification_method (str): Stratification method. Can be "quantile" or "sobol".
        importance_method (str): Importance method. Can be "sobol" or "quantile".
        seed (int): Seed for the random number generator.
        device (str): Device to use for the computation. Can be "cpu" or "cuda".
        dtype (str): Datatype to use for the computation. Can be "float32" or "float64".
    """

    def __init__(
        self,
        integration_domain=None,
        n_dim=None,
        n_eval=10000,
        batch_size=1000,
        n_stratification=1000,
        n_refinements=5,
        n_importance=1000,
        stratification_method="quantile",
        importance_method="sobol",
        seed=0,
        device="cpu",
        dtype="float64",
    ):

        self.vegas_stratification = VEGASStratification(
            n_stratification=n_stratification,
            n_refinements=n_refinements,
            stratification_method=stratification_method,
            importance_method=importance_method,
            seed=seed,
            device=device,
            dtype=dtype,
        )

        self.vegas_map = VEGASMap(
            n_eval=n_eval,
            batch_size=batch_size,
            n_importance=n_importance,
            seed=seed,
            device=device,
            dtype=dtype,
        )

        self.integration_domain = _setup_integration_domain(
            integration_domain, n_dim, device, dtype
        )

        self.seed = seed
        self.device = device
        self.dtype = dtype

        self.rng = RNG(seed=seed, device=device, dtype=dtype)

        self._backend = infer_backend(self.integration_domain)
        self._dim = self._get_dim(self.integration_domain)
        self._n_eval = n_eval

        self._n_stratification = n_stratification
        self._n_refinements = n_refinements
        self._n_importance = n_importance
        self._stratification_method = stratification_method
        self._importance_method = importance_method

        self._batch_size = batch_size
        self._n_batches = int(self._n_eval / self._batch_size)

        self._n_samples = self._n_stratification * self._n_refinements
        self._n_samples_per_batch = int(self._n_samples / self._n_batches)

        self._n_importance_samples = self._n_importance * self._n_refinements

        self._n_importance_samples_per_batch = int(


        """Integrate a function using VEGAS.

        Args:
            f (callable): Function to integrate.
            n_eval (int): Number of evaluations.
            batch_size (int): Number of function evaluations per batch.
            n_stratification (int): Number of stratification samples.
            n_refinements (int): Number of refinement iterations.
            n_importance (int): Number of importance samples.
            stratification_method (str): Stratification method. Can be "quantile" or "sobol".
            importance_method (str): Importance method. Can be "sobol" or "quantile".
        Returns:
            float: Integral estimate.
            float: Integral uncertainty estimate.
        """

if __name__ == "__main__":
    import torchquad as tq

    def f(x):
        return x[:, 0] ** 2 + x[:, 1] ** 2

    integrator = tq.VEGAS(n_eval=10000, n_stratification=1000, n_refinements=5)
    integral, error = integrator.integrate(f)
    print(integral, error)
    for i in range(10):
        integral, error = integrator.integrate(f)
        print(integral, error)
        # Stratification
        stratification_samples = self.vegas_stratification.get_samples(
            self.integration_domain, self._n_samples
        )
        stratification_weights = self.vegas_stratification.get_weights(
            self.integration_domain, stratification_samples
        )

        # Map
        map_samples, map_weights = self.vegas_map.get_samples_and_weights(
            f,
            self.integration_domain,
            stratification_samples,
            stratification_weights,
            self._n_importance_samples,
        )

        # Integrate
        integral = self._integrate(f, map_samples, map_weights)

        return integral

    def _integrate(self, f, samples, weights):
        """Integrate a function using VEGAS.

        Args:
            f (callable): Function to integrate.
            samples (backend tensor): Samples to use for the integration.
            weights (backend tensor): Weights to use for the integration.
        Returns:
            float: Integral estimate.
            float: Integral uncertainty estimate.
        """

        # Compute the integral
        integral = self._backend.sum(f(samples) * weights, axis=0)
        integral /= self._n_samples

        # Compute the integral uncertainty
        integral_uncertainty = self._backend.sum(
            (f(samples) - integral) ** 2 * weights, axis=0
        )
        integral_uncertainty /= self._n_samples
        integral_uncertainty = self._backend.sqrt(integral_uncertainty)

        return integral, integral_uncertainty
        # Set default values for the input parameters
        if n_eval is None:
            n_eval = self.n_eval
        if batch_size is None:
            batch_size = self.batch_size
        if n_stratification is None:
            n_stratification = self.n_stratification
        if n_refinements is None:
            n_refinements = self.n_refinements
        if n_importance is None:
            n_importance = self.n_importance
        if stratification_method is None:
            stratification_method = self.stratification_method
        if importance_method is None:
            importance_method = self.importance_method

        # Set up the integration domain
        integration_domain = _setup_integration_domain(
            self.integration_domain, self.n_dim, self.backend
        )

        # Set up the random number generator
        rng = RNG(self.backend, self.dtype, self.device)
        rng.set_seed(self.seed)

        # Set up the stratification map
        vegas_map = VEGASMap(
            self.backend,
            self.dtype,
            self.device,
            self.n_dim,
            n_stratification,
            stratification_method,
        )

        # Set up the importance map
        importance_map = VEGASMap(
            self.backend,
            self.dtype,
            self.device,
            self.n_dim,
            n_importance,
            importance_method,
        )

        # Initialize the weights
        weights = torch.ones(n_stratification, dtype=self.dtype, device=self.device)

        # Initialize the importance weights
        importance_weights = torch.ones(
            n_importance, dtype=self.dtype, device=self.device
        )

        # Initialize the integral estimate
        integral = torch
        """Integrate a function.

        Args:
            f (callable): Function to integrate. Must take backend tensors as input.
            n_eval (int): Number of evaluations.
            batch_size (int): Number of function evaluations per batch.
            n_stratification (int): Number of stratification samples.
            n_refinements (int): Number of refinement iterations.
            n_importance (int): Number of importance samples.
            stratification_method (str): Stratification method. Can be "quantile" or "sobol".
            importance_method (str): Importance method. Can be "sobol" or "quantile".

        Returns:
            float or backend tensor: Integral.
        """
        if n_eval is None:
            n_eval = self.n_eval
        if batch_size is None:
            batch_size = self.batch_size
        if n_stratification is None:
            n_stratification = self.n_stratification
        if n_refinements is None:
            n_refinements = self.n_refinements
        if n_importance is None:
            n_importance = self.n_importance
        if stratification_method is None:
            stratification_method = self.stratification_method
        if importance_method is None:
            importance_method = self.importance_method

        # Initialize stratification
        stratification = self.vegas_stratification.initialize_stratification(
            f=f,
            n_eval=n_eval,
            batch_size=batch_size,
            n_stratification=n_stratification,
            n_refinements=n_refinements,
            n_importance=n_importance,
            stratification_method=stratification_method,
            importance_method=importance_method,
        )

        # Initialize refinement
        refinement = self.vegas_stratification.initialize_refinement(
            stratification=stratification,
            n_refinements=n_refinements,
            n_importance=n_importance,
            importance_method=importance_method,
        )

        # Initialize importance
        importance = self.vegas_stratification.initialize_importance(
            refinement=refinement,
            n_importance=n_importance,
            importance_method=importance_method,
        )

        # Initialize importance
        stratification = self.vegas_stratification.initialize_stratification(
            f=f,
            n_eval=n_eval,
            batch_size=batch_size,
            n_stratification=n_stratification,
            n_refinements=n_refinements,
            n_importance=n_importance,
            stratification_method=stratification_method,
            importance_method=importance_method,
        )

        # Initialize refinement
        refinement = self.vegas_stratification.initialize_refinement(

            stratification=stratification,
            n_refinements=n_refinements,
            n_importance=n_importance,

            importance_method=importance_method,
        )






class VEGAS(BaseIntegrator):
    """VEGAS Enhanced. Refer to https://arxiv.org/abs/2009.05112 .
    Implementation inspired by https://github.com/ycwu1030/CIGAR/ .
    EQ <n> refers to equation <n> in the above paper.
    JAX and Tensorflow are unsupported.
    For Tensorflow there exists a VEGAS+ implementation called VegasFlow: https://github.com/N3PDF/vegasflow
    """

    # This VEGAS+ implementation uses many indexed assignments and methods
    # which change member variables. To support JAX and Tensorflow with a
    # reasonable performance, a lot of code would need to be rewritten to be
    # trace-compilable, which may in turn deteriorate the performance and/or
    # accuracy with PyTorch.

    def __init__(self):
        super().__init__()

    def integrate(
        self,
        fn,
        dim,
        N=10000,
        integration_domain=None,
        seed=None,
        rng=None,
        use_grid_improve=True,
        eps_rel=0,
        eps_abs=0,
        max_iterations=20,
        use_warmup=True,
        backend=None,
    ):
        """Integrates the passed function on the passed domain using VEGAS.

        If the integrand output is far away from zero, i.e. lies within [b, b+c] for a constant b with large absolute value and small constant c, VEGAS does not adapt well to the integrand. Shifting the integrand so that it is close to zero may improve the accuracy of the calculated integral in this case.

        Args:
            fn (func): The function to integrate over.
            dim (int): Dimensionality of the function to integrate.
            N (int, optional): Approximate maximum number of function evaluations to use for the integration. This value can be exceeded if the vegas stratification distributes evaluations per hypercube very unevenly. Defaults to 10000.
            integration_domain (list, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim.
            seed (int, optional): Random number generation seed for the sampling point creation; only set if provided. Defaults to None.
            rng (RNG, optional): An initialised RNG; this can be used as alternative to the seed argument and to avoid problems with integrand functions which reset PyTorch's RNG seed.
            use_grid_improve (bool, optional): If True, improve the vegas map after each iteration. Defaults to True.
            eps_rel (float, optional): Relative error to abort at. Defaults to 0.
            eps_abs (float, optional): Absolute error to abort at. Defaults to 0.
            max_iterations (int, optional): Maximum number of vegas iterations to perform. The number of performed iterations is usually lower than this value because the number of sample points per iteration increases every fifth iteration. Defaults to 20.
            use_warmup (bool, optional): If True, execute a warmup to initialize the vegas map. Defaults to True.
            backend (string, optional): Numerical backend. This argument is ignored if the backend can be inferred from integration_domain. "jax" and "tensorflow" are unsupported. Defaults to the backend from the latest call to set_up_backend or "torch" for backwards compatibility.

        Raises:
            ValueError: If the integration_domain or backend argument is invalid

        Returns:
            backend-specific float: Integral value
        """

        self._check_inputs(dim=dim, N=N, integration_domain=integration_domain)
        logger.debug(
            "\n VEGAS integrating a "
            + str(dim)
            + "-dimensional fn with "
            + str(N)
            + " points over "
            + str(integration_domain)
            + "\n",
        )

        self._dim = dim
        self._nr_of_fevals = 0
        self._max_iterations = max_iterations
        self._eps_rel = eps_rel
        self._eps_abs = eps_abs
        self.use_grid_improve = use_grid_improve
        self.N = N
        # Determine the number of evaluations per iteration
        self._starting_N = N // (self._max_iterations + 5)
        self._N_increment = N // (self._max_iterations + 5)
        self._fn = fn
        integration_domain = _setup_integration_domain(dim, integration_domain, backend)
        self.backend = infer_backend(integration_domain)
        if self.backend in ["jax", "tensorflow"]:
            raise ValueError(f"Unsupported numerical backend: {self.backend}")
        self.dtype = integration_domain.dtype
        if rng is None:
            rng = RNG(backend=self.backend, seed=seed)
        elif seed is not None:
            raise ValueError("seed and rng cannot both be passed")
        self.rng = rng

        # Transform the integrand into the [0,1]^dim domain
        domain_starts = integration_domain[:, 0]
        domain_sizes = integration_domain[:, 1] - domain_starts
        domain_volume = anp.prod(domain_sizes)

        def transformed_integrand(x):
            return fn(x * domain_sizes + domain_starts) * domain_volume

        self._fn = transformed_integrand

        # Initialize the adaptive VEGAS map,
        # Note that a larger number of intervals may lead to problems if only few evals are allowed
        # Paper section II B
        N_intervals = max(2, self._N_increment // 10)  # for small N intervals set 2
        self.map = VEGASMap(N_intervals, self._dim, self.backend, self.dtype)

        # Initialize VEGAS' stratification
        # Paper section III
        self.strat = VEGASStratification(
            self._N_increment,
            dim=self._dim,
            rng=self.rng,
            backend=self.backend

            dtype=self.dtype,
        )

        logger.debug("Starting VEGAS")

        for iteration in range(self._max_iterations):
            # Execute warmup
            if iteration == 0 and use_warmup:
                self._warmup()

            # Execute iteration
            self._iterate()

            # Check if the error is small enough
            if self._error_is_small():
                break

        # Return the integral value
        return self._integral_value()

        self.results = []  # contains integration results per iteration
        if self._nr_of_fevals > self.N:
            logger.warning(
                "The number of function evaluations exceeded the maximum number of evaluations."
            )

    def _warmup(self):
        """Executes a warmup to initialize the vegas map."""
        logger.debug("Starting warmup")
        # Execute warmup
        warmup_N = self._starting_N // 10
        warmup_points = self.rng.uniform((warmup_N, self._dim))
        warmup_fn_values = self._fn(warmup_points)
        self._nr_of_fevals += warmup_N
        self.map.update(warmup_points, warmup_fn_values)
        logger.debug("Finished warmup")

    def _iterate(self, use_warmup=None):

        """Executes one iteration of VEGAS."""
        logger.debug("Starting iteration")
        # Execute stratification
        strat_points = self.strat(self.map)
        strat_fn_values = self._fn(strat_points)
        self._nr_of_fevals += self._N_increment
        # Update the map
        self.map.update(strat_points, strat_fn_values)
        self.sigma2 = []  # contains variance per iteration

        self.it = 0  # iteration

        if use_warmup:  # warmup the adaptive map
            self._warmup_grid(5, self._starting_N // 5)

        # Main loop
        while True:
            self.it = self.it + 1
            self.results.append(0)
            self.sigma2.append(0)

            # Compute current iteration
            acc = self._run_iteration()

            logger.info(
                f"Iteration {self.it}, Acc={acc:.4e}, Result={self.results[-1]:.4e},neval={self._nr_of_fevals}"
            )

            if self._check_abort_conditions():
                break

        logger.info(
            f"Computed integral after {self._nr_of_fevals} evals was {self._get_result():.8e}."
        )
        return self._get_result()

    def _check_abort_conditions(self):
        """Test if VEGAS should execute more iterations or stop,
        and every fifth iteration reset the sample integral results
        and adjust the number of evaluations per iteration

        Returns:
            Bool: True iff VEGAS should abort
        """
        # Abort only every fifth iteration
        if self.it % 5 > 0:
            return False

        # Abort conditions depending on achieved errors
        res_abs = anp.abs(self._get_result())
        err = self._get_error()
        chi2 = self._get_chisq()
        logger.debug(f"Iteration {self.it},Chi2={chi2:.4e}")
        if (
            err <= self._eps_rel * res_abs or err <= self._eps_abs
        ) and chi2 / 5.0 < 1.0:
            return True

        # Adjust number of evals if Chi square indicates instability
        # EQ 32
        if chi2 / 5.0 < 1.0:
            # Use more points in the next iterations to reduce the
            # relative error
            if res_abs == 0.0:
                self._starting_N += self._N_increment
            else:
                acc = err / res_abs
                self._starting_N = min(
                    self._starting_N + self._N_increment,
                    int(self._starting_N * anp.sqrt(acc / (self._eps_rel + 1e-8))),
                )
        elif chi2 / 5.0 > 1.0:
            # Use more points in the next iterations because of instability
            self._starting_N += self._N_increment

        # Abort if the next 5 iterations would use too many function
        # evaluations
        if self._nr_of_fevals + self._starting_N * 5 > self.N:
            return True

        # Abort if the maximum nuber of iterations is reached
        if self.it + 5 > self._max_iterations:
            return True

        self.results = []  # reset sample results
        self.sigma2 = []  # reset sample results
        return False

    def _warmup_grid(self, warmup_N_it=5, N_samples=1000):
        """This function warms up the adaptive map of VEGAS over some iterations and samples.

        Args:
            warmup_N_it (int, optional): Number of warmup iterations. Defaults to 5.
            N_samples (int, optional): Number of samples per warmup iteration. Defaults to 1000.
        """
        logger.debug(
            f"Running Map Warmup with warmup_N_it={warmup_N_it}, N_samples={N_samples}..."
        )

        alpha_start = 0.5  # initial alpha value
        # TODO in the original paper this is adjusted over time
        self.alpha = alpha_start

        # Warmup
        logger.debug(
            "|  Iter  |    N_Eval    |     Result     |      Error     |    Acc        | Total Evals"
        )
        for warmup_iter in range(warmup_N_it):
            self.results.append(0)
            self.sigma2.append(0)
            jf = 0  # jacobians * function
            jf2 = 0

            # Sample points yrnd and transformed sample points x
            # Multiplying by 0.99999999 as the edge case of y=1 leads to an error
            yrnd = (
                self.rng.uniform(size=[N_samples, self._dim], dtype=self.dtype)
                * 0.999999
            )
            x = self.map.get_X(yrnd)
            f_eval = self._eval(x).squeeze()
            jac = self.map.get_Jac(yrnd)
            jf_vec = f_eval * jac
            jf_vec2 = jf_vec**2
            if self.backend == "torch":
                jf_vec2 = jf_vec2.detach()
            self.map.accumulate_weight(yrnd, jf_vec2)  # update map weights
            jf = jf_vec.sum()
            jf2 = jf_vec2.sum()

            ih = jf / N_samples  # integral in this step
            sig2 = jf2 / N_samples - pow(jf / N_samples, 2)  # estimated variance
            if self.backend == "torch":
                sig2 = sig2.detach()
            # Sometimes rounding errors produce negative values very close to 0
            sig2 = anp.abs(sig2)
            self.results[-1] += ih  # store results
            self.sigma2[-1] += sig2 / N_samples  # store results
            self.map.update_map()  # adapt the map
            # Estimate an accuracy for the logging
            acc = anp.sqrt(self.sigma2[-1])
            if self.results[-1] != 0.0:
                acc = acc / anp.abs(self.results[-1])
            logger.debug(
                f"|\t{warmup_iter}|         {N_samples}|  {self.results[-1]:5e}  |  {self.sigma2[-1]:5e}  |  {acc:4e}%| {self._nr_of_fevals}"
            )

        self.results.clear()
        self.sigma2.clear()

    def _run_iteration(self):
        """Runs one iteration of VEGAS including stratification and updates the VEGAS map if use_grid_improve is set.

        Returns:
            backend-specific float: Estimated accuracy.
        """
        neval = self.strat.get_NH(self._starting_N)  # Evals per strat cube

        # Stratified sampling points y and transformed sample points x
        y = self.strat.get_Y(neval)
        x = self.map.get_X(y)  # transform, EQ 8+9

        # TODO: for batch 把这里拆开
        # Evaluate the integrand and remember the number of evaluations
        f_eval = self._eval(x).squeeze()

        jac = self.map.get_Jac(y)  # compute jacobian
        jf_vec = f_eval * jac  # precompute product once
        jf_vec2 = jf_vec**2
        if self.backend == "torch":
            jf_vec2 = jf_vec2.detach()

        if self.use_grid_improve:  # if adaptive map is used, acc weight
            self.map.accumulate_weight(y, jf_vec2)  # EQ 25
        jf, jf2 = self.strat.accumulate_weight(neval, jf_vec)  # update strat

        neval_inverse = 1.0 / astype(neval, y.dtype)
        ih = jf * (neval_inverse * self.strat.V_cubes)  # Compute integral per cube

        # Collect results
        sig2 = jf2 * neval_inverse * (self.strat.V_cubes**2) - pow(ih, 2)
        if self.backend == "torch":
            sig2 = sig2.detach()
        # Sometimes rounding errors produce negative values very close to 0
        sig2 = anp.abs(sig2)
        self.results[-1] = ih.sum()  # store results
        self.sigma2[-1] = (sig2 * neval_inverse).sum()

        if self.use_grid_improve:  # if on, update adaptive map
            logger.debug("Running grid improvement")
            self.map.update_map()
        self.strat.update_DH()  # update stratification

        # Estimate an accuracy for the logging
        acc = anp.sqrt(self.sigma2[-1])
        if self.results[-1] != 0.0:
            acc = acc / anp.abs(self.results[-1])

        return acc

    # Helper funcs
    def _get_result(self):
        """Computes mean of results to estimate integral, EQ 30.

        Returns:
            backend-specific float: Estimated integral.
        """
        if any(sig2 == 0.0 for sig2 in self.sigma2):
            # If at least one variance is 0, return the mean result
            res = sum(self.results) / len(self.results)
        else:
            res_num = sum(res / sig2 for res, sig2 in zip(self.results, self.sigma2))
            res_den = sum(1.0 / sig2 for sig2 in self.sigma2)
            res = res_num / res_den
        if self.backend == "numpy" and res.dtype != self.results[0].dtype:
            # NumPy automatically casts float32 to float64 in the above
            # calculations
            res = astype(res, self.results[0].dtype)
        return res

    def _get_error(self):
        """Estimates error from variance , EQ 31.

        Returns:
            backend-specific float: Estimated error.

        """
        # Skip variances which are zero and return a backend-specific float
        res = sum(1.0 / sig2 for sig2 in self.sigma2 if sig2 != 0.0)
        return self.sigma2[0] if res == 0 else 1.0 / anp.sqrt(res)

    def _get_chisq(self):
        """Computes chi square from estimated integral and variance, EQ 32.

        Returns:
            backend-specific float: Chi squared.
        """
        I_final = self._get_result()
        return sum(
            (
                (res - I_final) ** 2 / sig2
                for res, sig2 in zip(self.results, self.sigma2)
                if res != I_final
            ),
            start=self.results[0] * 0.0,
        )
