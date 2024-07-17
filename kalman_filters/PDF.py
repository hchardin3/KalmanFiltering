import numpy as np
import math
from numpy.linalg import det, inv, slogdet, cholesky
from scipy.stats import multivariate_normal
from scipy.integrate import nquad

class ProbabilityDensityFunction:
    def __init__(self, pdf, n_dim: int, pdf_bounds: list[tuple[float, float]]|None = None, sampling_method: str = "default", mean: np.ndarray = None, covariance_matrix: np.ndarray = None) -> None:
        """
        A Probability Density Function is assumed to cover the vector space R^n_dim, and to have the regular properties of the pdf.

        Parameters:
            pdf (callable): A probability density function over R^n_dim. Takes a vector as an input and returns a probability. Is assumed to always be positive, and to have an integral over R^n_dim of 1.
            n_dim (int): The dimension of the vector space.
            pdf_bounds (list of tuples): The scalar limits of the definition space of the pdf in R^n_dim. Should be of length n_dim.
            sampling_method (str): Specifies the method to be used for sampling. Must be one of ["default", "gaussian"].
            mean (np.ndarray): The mean of the distribution, if known.
            covariance_matrix (np.ndarray): The covariance matrix of the distribution, if known.
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

        samples = np.zeros((n_points, self.n_dim))
        for i in range(n_points):
            while True:
                point = np.array([np.random.uniform(low, high) for low, high in space_bounds])
                prob = self.pdf(point)
                if np.random.rand() < prob:
                    samples[i] = point
                    break
        return samples
    
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

class MultivariateGaussianPDF(ProbabilityDensityFunction):
    def __init__(self, mean: np.ndarray|None = None, covariance_matrix: np.ndarray|None = None) -> None:
        if mean is None:
            mean = np.zeros(2)  # Default to 2D for simplicity if not specified
        if covariance_matrix is None:
            covariance_matrix = np.eye(len(mean))  # Identity matrix of dimension len(mean)

        n_dim = mean.size

        # Define the pdf function using scipy's multivariate normal distribution
        scipy_mv_gaussian = multivariate_normal(mean=mean, cov=covariance_matrix)

        def pdf(x: np.ndarray):
            return scipy_mv_gaussian.pdf(x)

        super().__init__(pdf, n_dim, sampling_method="gaussian", mean=mean, covariance_matrix=covariance_matrix)
    
    def change_gaussian_law(self, new_mean: np.ndarray|None = None, new_covariance_matrix: np.ndarray|None = None):
        """
        Allow to update the gaussian law of this PDF by simply providing new mean vector and covariance matrix.

        Parameters:
            new_mean (np.ndarray): The new mean vector.
            new_covariance_matrix (np.ndarray): The new covariance matrix.
        """
        if new_mean is not None or new_covariance_matrix is not None:
            if new_mean is not None and new_mean.size == self.n_dim:
                self.mean = new_mean
            
            if new_covariance_matrix is not None and new_covariance_matrix.shape[0] == self.n_dim:
                self.covariance_matrix = new_covariance_matrix
            
            def new_pdf(x: np.ndarray):
                # Calculate the normalizing constant
                denom = np.sqrt((2 * np.pi) ** self.n_dim * det(self.covariance_matrix))
                # Calculate the exponent term
                x_m = x - self.mean
                exponent = -0.5 * np.dot(x_m.T, np.dot(inv(self.covariance_matrix), x_m))
                return np.exp(exponent) / denom
            
            self.pdf = new_pdf

class MultivariateUniformPDF(ProbabilityDensityFunction):
    def __init__(self, bounds: list[tuple[float, float]]) -> None:
        """
        Creates a Uniform Probability Distribution over the specified bounds.

        Parameters:
            bounds (list of tuples of shape [lim1, lim2] with lim2 > lim1): The bounds of the uniform pdf for each dimension.
        """
        n_dim = len(bounds)
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
        Creates a Multivariate Exponential Probability Distribution with given scales for each dimension.

        Parameters:
            scales (np.ndarray): Scale parameters for each dimension.
        """
        n_dim = scales.size
        mean = scales
        covariance_matrix = np.diag(scales**2)

        def pdf(x: np.ndarray):
            if np.any(x < 0):
                return 0.0
            return np.prod([(1/scale) * np.exp(-x[i]/scale) for i, scale in enumerate(scales)])

        super().__init__(pdf, n_dim, pdf_bounds=[(0, 10 * scales[i]) for i in range(n_dim)], mean=mean, covariance_matrix=covariance_matrix)

class MonovariateExponentialPDF(ProbabilityDensityFunction):
    def __init__(self, scale: float = 1.0) -> None:
        """
        Creates an Exponential Probability Distribution with a given scale (lambda).

        Parameters:
            scale (float): The scale parameter (1/lambda) of the exponential distribution. Default is 1.0.
        """
        mean = np.array([scale])
        covariance_matrix = np.array([[scale**2]])

        def pdf(x: np.ndarray):
            return (1/scale) * np.exp(-x/scale) if x >= 0 else 0.0

        super().__init__(pdf, 1, pdf_bounds=[(0, 10 * scale)], mean=mean, covariance_matrix=covariance_matrix)

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

        def pdf(x: np.ndarray):
            if x >= 0:
                return (x**(shape-1) * np.exp(-x/scale)) / (gamma_func(shape) * scale**shape)
            else:
                return 0.0

        super().__init__(pdf, 1, pdf_bounds=[(0, 10 * scale)], mean=mean, covariance_matrix=covariance_matrix)

