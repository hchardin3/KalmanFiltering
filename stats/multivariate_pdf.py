from stats.probability_density import ProbabilityDensityFunction
import numpy as np
from scipy.stats import multivariate_normal

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
            raise(ValueError("Please use univariate_pdf.UnivariateGaussianPDF for gaussian distribution laws on R"))
        
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
            raise(ValueError("Please use univariate_pdf.UnivariateUniformPDF for uniform pdfs of size 1"))
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
        Initializes a multivariate exponential distribution defined using specified scales for each dimension.

        Parameters:
            scales (np.ndarray): A vector used to define the scale of the exponential law over each dimension.

        This class is suitable for defining exponential distributions in higher dimensions where each dimension can have a different scale.
        """
        n_dim = scales.size
        mean = scales
        covariance_matrix = np.diag(scales**2)

        def pdf(x: np.ndarray):
            if np.any(x < 0):
                return 0.0
            return np.prod([(1/scale) * np.exp(-x[i]/scale) for i, scale in enumerate(scales)])

        super().__init__(pdf, n_dim, pdf_bounds=[(0, 100 * scales[i]) for i in range(n_dim)], mean=mean, covariance_matrix=covariance_matrix)
