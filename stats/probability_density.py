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
        