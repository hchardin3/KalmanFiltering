"""
This script is used to define Probability Density Functions (PDFs) as Python objects.
"""

import numpy as np
import math
from scipy.stats import multivariate_normal
from scipy.integrate import nquad

class ProbabilityDensityFunction:
    def __init__(self, pdf, n_dim: int, pdf_bounds: list[tuple[float, float]]|None = None, sampling_method: str = "default", mean: np.ndarray = None, covariance_matrix: np.ndarray = None):
        """
        Initializes a general Probability Density Function (PDF) object for defining and working with custom PDFs.

        Parameters:
            pdf (callable): A function that takes a vector (np.ndarray) and returns the probability density at that point.
            n_dim (int): Dimensionality of the PDF (i.e., the number of random variables it represents).
            pdf_bounds (list of tuple of float, optional): Bounds for each dimension beyond which the PDF is considered negligible. Each tuple represents (min, max).
            sampling_method (str): Method used for sampling from the PDF. Supported methods: 'default' for custom sampling logic, 'gaussian' for Gaussian approximations.
            mean (np.ndarray, optional): Known mean of the distribution. Provided to avoid recomputation if already known.
            covariance_matrix (np.ndarray, optional): Known covariance matrix of the distribution. Provided to avoid recomputation if already known.

        The `pdf` function should integrate to 1 over its entire support and must be capable of handling inputs of shape (n_dim,).
        If `mean` and `covariance_matrix` are not provided, they will be estimated if possible.
        """
        self.pdf = pdf

        available_sampling_methods = ["default", "gaussian"]
        if sampling_method in available_sampling_methods:
            self.sampling_method = sampling_method
        else:
            raise(ValueError(f"sampling_method should be in {available_sampling_methods:}"))
        
        self.n_dim = n_dim
        if mean is not None:
            if mean.size == n_dim:
                self.mean = mean
            else:
                raise(ValueError("mean should be of size n_dim"))
            
        if covariance_matrix is not None:
            if covariance_matrix.shape[0] == n_dim and covariance_matrix.shape[1] == n_dim:
                self.covariance_matrix = covariance_matrix
            else:
                raise(ValueError("covariance_matrix should be a square array of size n_dim"))
        
        self.space_bounds = None

        if self.mean is None or self.covariance_matrix is None:
            if pdf_bounds is None:
                self.space_bounds = self.estimate_bounds()
            else:
                if len(pdf_bounds) != n_dim:
                    raise(ValueError("pdf_bounds should be of length n_dim"))
                else:
                    self.space_bounds = pdf_bounds
            
            if self.mean is None:
                self.mean = self.compute_mean_integral()
            if self.covariance_matrix is None:
                self.covariance_matrix = self.compute_covariance_integral()
        
        if self.space_bounds is None:
            if pdf_bounds is None:
                self.space_bounds = self.compute_space_bounds()
            else:
                if len(pdf_bounds) == n_dim: self.space_bounds = pdf_bounds
                else: raise(ValueError("pdf_bounds should be of length n_dim"))

    def evaluate(self, x):
        return self.pdf(x)
    
    def sample(self, n_points):
        if self.sampling_method == "default":
            return self.default_sampler(n_points)
        elif self.sampling_method == "gaussian":
            return self.gaussian_sampler(n_points)
        
    def default_sampler(self, n_points):
        if self.space_bounds is None:
            space_bounds = [(-10**6, 10**6) for _ in range(self.n_dim)]
        else:
            space_bounds = self.space_bounds
        
        if n_points > 1:
            samples = np.zeros((n_points, self.n_dim))
            for i in range(n_points):
                while True:
                    point = np.array([np.random.uniform(low, high) for low, high in space_bounds])
                    prob = self.pdf(point)
                    if np.random.rand() < prob:
                        samples[i] = point
                        break
            return samples
        else:
            while True:
                    point = np.array([np.random.uniform(low, high) for low, high in space_bounds])
                    prob = self.pdf(point)
                    if np.random.rand() < prob:
                        return point

    
    def gaussian_sampler(self, n_points):
        return multivariate_normal(mean=self.mean, cov=self.covariance_matrix).rvs(size=n_points)

    def compute_mean(self, samples):
        """
        Compute the mean of the samples.
        
        Parameters:
            samples: np.ndarray - Sampled points
        
        Returns:
            mean: np.ndarray - Computed mean
        """
        return np.mean(samples, axis=0)
    
    def compute_covariance(self, samples: np.ndarray):
        """
        Compute the covariance matrix of the samples.
        
        Parameters:
            samples: np.ndarray - Sampled points
        
        Returns:
            covariance_matrix: np.ndarray - Computed covariance matrix
        """
        return np.cov(samples.T, rowvar=False)
    
    def compute_mean_integral(self):
        """
        Compute the mean of the distribution using numerical integration.
        
        Returns:
            mean: np.ndarray - Computed mean
        """
        bounds = self.space_bounds

        def integrand(*args):
            x = np.array(args)
            return x * self.pdf(x)

        result = np.zeros(self.n_dim)
        for i in range(self.n_dim):
            integrand_i = lambda *args: integrand(*args)[i]
            result[i], _ = nquad(integrand_i, bounds)
        return result
    
    def compute_covariance_integral(self):
        """
        Compute the covariance matrix of the distribution using numerical integration.
        
        Returns:
            covariance_matrix: np.ndarray - Computed covariance matrix
        """
        bounds = self.space_bounds

        def integrand(*args):
            x = np.array(args)
            mu = self.mean
            return np.outer(x - mu, x - mu) * self.pdf(x)

        result = np.zeros((self.n_dim, self.n_dim))
        for i in range(self.n_dim):
            for j in range(self.n_dim):
                integrand_ij = lambda *args: integrand(*args)[i, j]
                result[i, j], _ = nquad(integrand_ij, bounds)
        return result
    
    def compute_space_bounds(self, bound_factor: int | float = 10):
        """
        Compute the space bounds based on the mean and covariance matrix.

        Parameters:
            bound_factor (int or float): A number to scale the bound.
        
        Returns:
            space_bounds: list of tuples - Computed bounds for each dimension
        """
        if self.n_dim == 1:
            x = np.sqrt(self.covariance_matrix)
            space_bounds = [(self.mean - x, self.mean + x)]
        else:
            max_eigenvalue = np.max(np.linalg.eigvals(self.covariance_matrix))
            x = np.abs(self.mean) * bound_factor * max_eigenvalue
            space_bounds = [(self.mean[i] - x[i], self.mean[i] + x[i]) for i in range(self.n_dim)]
        return space_bounds
    
    def estimate_bounds(self, initial_range=10**6, num_samples=1000):
        """
        Estimate integration bounds by sampling the PDF.

        Parameters:
        initial_range: float - Initial range to consider for sampling
        num_samples: int - Number of samples to draw for estimating bounds

        Returns:
        bounds: list of tuples - Estimated bounds for each dimension
        """
        samples = np.random.uniform(-initial_range, initial_range, (num_samples, self.n_dim))
        values = np.apply_along_axis(self.pdf, 1, samples)
        non_zero_samples = samples[values > 0]
        
        bounds = []
        if non_zero_samples.shape[0] == 0:
            raise ValueError("No non-zero samples found within initial range.")
        else:
            for dim in range(self.n_dim):
                lower_bound = np.percentile(non_zero_samples[:, dim], 2.5)
                upper_bound = np.percentile(non_zero_samples[:, dim], 97.5)
                bounds.append((lower_bound, upper_bound))
            return bounds

class UnivariateGaussianPDF(ProbabilityDensityFunction):
    def __init__(self, mean: float|int = 0, covariance: float|int|None = None):
        """
        Initializes a univariate Gaussian distribution with specified mean and variance.

        Parameters:
            mean (float|int): Mean of the Gaussian distribution.
            covariance (float|int, optional): Variance of the Gaussian distribution. Defaults to 1 if not specified.

        This class represents a Gaussian distribution for a single variable and leverages scipy's multivariate_normal for efficiency.
        """        
        if covariance is None:
            covariance = 1

        # Define the pdf function using scipy's multivariate normal distribution
        scipy_mv_gaussian = multivariate_normal(mean=mean, cov=covariance)

        def pdf(x):
            return scipy_mv_gaussian.pdf(x)
        
        scaling_factor = math.sqrt(covariance)
        
        bounds = [(mean - 100 * scaling_factor, mean + 100 * scaling_factor)]

        super().__init__(pdf, 1, pdf_bounds=bounds, sampling_method="gaussian", mean=np.array(mean), covariance_matrix=np.diag([covariance]))

class MultivariateGaussianPDF(ProbabilityDensityFunction):
    def __init__(self, mean: np.ndarray = np.zeros(2), covariance_matrix: np.ndarray|None = None):
        """
        Initializes a multivariate Gaussian distribution with specified mean and covariance matrix.

        Parameters:
            mean (np.ndarray): Mean vector of the Gaussian distribution.
            covariance_matrix (np.ndarray, optional): Covariance matrix of the Gaussian distribution. If not provided, defaults to the identity matrix scaled by the dimension of `mean`.

        Ensures that the covariance matrix is square and matches the dimensionality of the mean vector. Uses scipy's multivariate_normal for computations.
        """
        if len(mean.shape) != 1:
            raise(ValueError("mean should be a vector"))
        elif mean.size == 1:
            raise(ValueError("Please use UnivariateGaussianPDF for gaussian distribution laws on R"))
        
        if covariance_matrix is None:
            covariance_matrix = np.eye(mean.size)
        else:
            if len(covariance_matrix.shape) != 2 or covariance_matrix.shape[0] != covariance_matrix.shape[1]:
                raise(ValueError("covariance_matrix must be a square matrix"))
            
            if covariance_matrix.shape[0] != mean.size:
                raise(ValueError("covariance_matrix size should be equal to mean length"))

        n_dim = mean.size

        # Define the pdf function using scipy's multivariate normal distribution
        scipy_mv_gaussian = multivariate_normal(mean=mean, cov=covariance_matrix)

        def pdf(x: np.ndarray):
            return scipy_mv_gaussian.pdf(x)
        
        scaling_factor = np.sqrt(mean.size * np.linalg.norm(covariance_matrix))
        
        bounds = [(mean[i] - 100 * scaling_factor, mean[i] + 100 * scaling_factor) for i in range(mean.size)]

        super().__init__(pdf, n_dim, pdf_bounds=bounds, sampling_method="gaussian", mean=mean, covariance_matrix=covariance_matrix)

class UnivariateUniformPDF(ProbabilityDensityFunction):
    def __init__(self, lower_bound: int|float, upper_bound: int|float):
        """
        Initializes a uniform distribution over a specified range.

        Parameters:
            lower_bound (int|float): The lower bound of the distribution.
            upper_bound (int|float): The upper bound of the distribution. Must be strictly greater than `lower_bound`.

        This distribution assigns equal probability to all outcomes in the specified range and zero probability outside this range.
        """
        if upper_bound <= lower_bound:
            raise(ValueError("Please specify valid bounds"))
        
        volume = upper_bound - lower_bound
        mean = (lower_bound + upper_bound) / 2
        covariance_matrix = (upper_bound - lower_bound)**2 / 12

        def pdf(x: float):
            in_bounds = (x >= lower_bound and x <= upper_bound)
            return 1 / volume if in_bounds else 0.0
        
        bounds = [(lower_bound, upper_bound)]

        super().__init__(pdf, 1, pdf_bounds=bounds, mean=np.array(mean), covariance_matrix=np.diag([covariance_matrix]))

class MultivariateUniformPDF(ProbabilityDensityFunction):
    def __init__(self, bounds: list[tuple[float, float]]):
        """
        Initializes a multivariate uniform distribution defined over specified bounds for each dimension.

        Parameters:
            bounds (list of tuples of float): Bounds for each dimension in the format [(lower1, upper1), (lower2, upper2), ...]. Each tuple specifies the range for that dimension.

        This class is suitable for defining uniform distributions in higher dimensions where each dimension can have different bounds.
        """
        n_dim = len(bounds)

        if n_dim == 1:
            raise(ValueError("Please use UnivariateUniformPDF for uniform pdfs of size 1"))
        else:
            for i in range(n_dim):
                if len(bounds[i]) != 2:
                    raise(ValueError(f"bound of index {i} is uncorrectly defined: should only include two numbers."))
                elif bounds[i][0] >= bounds[i][1]:
                    raise(ValueError(f"bound of index {i} is uncorrectly defined: lower bound should be strictly smaller than upper bound."))

        volume = np.prod([bound[1] - bound[0] for bound in bounds])
        mean = np.array([(bound[0] + bound[1]) / 2 for bound in bounds])
        covariance_matrix = np.diag([(bound[1] - bound[0])**2 / 12 for bound in bounds])

        def pdf(x: np.ndarray):
            in_bounds = np.all([bounds[i][0] <= x[i] <= bounds[i][1] for i in range(n_dim)])
            return 1 / volume if in_bounds else 0.0

        super().__init__(pdf, n_dim, pdf_bounds=bounds, mean=mean, covariance_matrix=covariance_matrix)

class MultivariateExponentialPDF(ProbabilityDensityFunction):
    def __init__(self, scales: np.ndarray) -> None:
        """
        Initializes a multivariate uniform distribution defined over specified bounds for each dimension.

        Parameters:
            bounds (list of tuples of float): Bounds for each dimension in the format [(lower1, upper1), (lower2, upper2), ...].
                Each tuple specifies the range for that dimension.

        This class is suitable for defining uniform distributions in higher dimensions where each dimension can have different bounds.
        """
        n_dim = scales.size
        mean = scales
        covariance_matrix = np.diag(scales**2)

        def pdf(x: np.ndarray):
            if np.any(x < 0):
                return 0.0
            return np.prod([(1/scale) * np.exp(-x[i]/scale) for i, scale in enumerate(scales)])

        super().__init__(pdf, n_dim, pdf_bounds=[(0, 100 * scales[i]) for i in range(n_dim)], mean=mean, covariance_matrix=covariance_matrix)

class UnivariateExponentialPDF(ProbabilityDensityFunction):
    def __init__(self, scale: float = 1.0):
        """
        Initializes a univariate exponential distribution with a specified scale parameter.

        Parameters:
            scale (float): The scale parameter of the exponential distribution, often denoted as lambda^-1.

        Represents the time between events in a Poisson point process, i.e., the time until the next event occurs.
        """
        mean = np.array([scale])
        covariance_matrix = np.array([[scale**2]])

        def pdf(x: np.ndarray):
            return (1/scale) * np.exp(-x/scale) if x >= 0 else 0.0

        super().__init__(pdf, 1, pdf_bounds=[(0, 100 * scale)], mean=mean, covariance_matrix=covariance_matrix)

class UnivariateBetaPDF(ProbabilityDensityFunction):
    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        """
        Creates a Beta Probability Distribution with given alpha and beta parameters.

        Parameters:
            alpha (float): The alpha parameter of the beta distribution. Default is 1.0.
            beta (float): The beta parameter of the beta distribution. Default is 1.0.
        """
        from scipy.special import beta as beta_func

        mean = np.array([alpha / (alpha + beta)])
        variance = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
        covariance_matrix = np.array([[variance]])

        def pdf(x: np.ndarray):
            if 0 <= x <= 1:
                return (x**(alpha-1) * (1-x)**(beta-1)) / beta_func(alpha, beta)
            else:
                return 0.0

        super().__init__(pdf, 1, pdf_bounds=[(0, 1)], mean=mean, covariance_matrix=covariance_matrix)

class UnivariateGammaPDF(ProbabilityDensityFunction):
    def __init__(self, shape: float = 1.0, scale: float = 1.0) -> None:
        """
        Creates a Gamma Probability Distribution with given shape (k) and scale (theta) parameters.

        Parameters:
            shape (float): The shape parameter (k) of the gamma distribution. Default is 1.0.
            scale (float): The scale parameter (theta) of the gamma distribution. Default is 1.0.
        """
        from scipy.special import gamma as gamma_func

        mean = np.array([shape * scale])
        variance = shape * (scale ** 2)
        covariance_matrix = np.array([[variance]])

        def pdf(x):
            if x >= 0:
                return (x**(shape-1) * np.exp(-x/scale)) / (gamma_func(shape) * scale**shape)
            else:
                return 0.0

        super().__init__(pdf, 1, pdf_bounds=[(0, 100 * scale)], mean=mean, covariance_matrix=covariance_matrix)

