from stats.probability_density import ProbabilityDensityFunction
import numpy as np
import math
from scipy.stats import multivariate_normal

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
            raise(ValueError("Please specify valid bounds, ie lower_bound < upper_bound"))
        
        volume = upper_bound - lower_bound
        mean = (lower_bound + upper_bound) / 2
        covariance_matrix = (upper_bound - lower_bound)**2 / 12

        def pdf(x: float):
            in_bounds = (x >= lower_bound and x <= upper_bound)
            return 1 / volume if in_bounds else 0.0
        
        bounds = [(lower_bound, upper_bound)]

        super().__init__(pdf, 1, pdf_bounds=bounds, mean=np.array(mean), covariance_matrix=np.diag([covariance_matrix]))

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