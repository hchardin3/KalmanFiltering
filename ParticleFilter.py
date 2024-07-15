import numpy as np
from numpy.linalg import det, inv, slogdet
from scipy.stats import multivariate_normal

class ProbabilityDensityFunction:
    def __init__(self, pdf, n_dim: int = 1, mean:np.ndarray = None, covariance_matrix:np.ndarray = None) -> None:
        """
        A Probability Density Function is assumed to cover the vector space R^n_dim, and to have the regular properties of the pdf.

        Parameters:
            pdf (callable): A probability density function over R^n_dim. Takes a vector as an input and return a probability. Is assumed to always be positive, and has an integral of 1.
            n_dim (int): The dimension of the space. 
        """
        self.pdf = pdf
        self.n_dim = n_dim

        self.mean = mean
        self.covariance_matrix = covariance_matrix
    
    def evaluate(self, x):
        return self.pdf(x)
    
    def sample(self, n_points, space_bounds = None):
        """
        Sample N random points according to the probability density function.
        
        Parameters:
        n_points: int - Number of points to sample
        space_bounds: list of tuples - Bounds for each dimension [(min1, max1), (min2, max2), ...]
        
        Returns:
        samples: np.ndarray - Sampled points
        """
        if space_bounds is None:
            space_bounds = ((-10**9, 10**9) for _ in range(self.n_dim))

        samples = np.zeros((n_points, self.n_dim))
        for i in range(n_points):
            while True:
                point = np.array([np.random.uniform(low, high) for low, high in space_bounds])
                prob = self.pdf(point)
                if np.random.rand() < prob:
                    samples[i] = point
                    break
        return samples

class GaussianPDF(ProbabilityDensityFunction):
    def __init__(self, mean: np.ndarray = None, covariance_matrix: np.ndarray = None) -> None:
        if mean is None:
            mean = np.zeros(2)  # Default to 2D for simplicity if not specified
        if covariance_matrix is None:
            covariance_matrix = np.eye(len(mean))  # Identity matrix of dimension len(mean)
        
        self.mean = mean
        self.covariance_matrix = covariance_matrix
        self.n_dim = mean.size

        # Define the pdf function using a closure to capture mean and covariance_matrix
        def pdf(x: np.ndarray):
            # Calculate the normalizing constant
            denom = np.sqrt((2 * np.pi) ** self.n_dim * det(covariance_matrix))
            # Calculate the exponent term
            x_m = x - mean
            exponent = -0.5 * np.dot(x_m.T, np.dot(inv(covariance_matrix), x_m))
            return np.exp(exponent) / denom

        super().__init__(pdf, self.n_dim, self.mean, self.covariance_matrix)

class ParticleFilter:
    def __init__(self, f, h, N_particles: int, x0: np.ndarray, P0: np.ndarray, dynamics_noise_pdf: ProbabilityDensityFunction, measurement_noise_pdf: ProbabilityDensityFunction, x0_pdf: ProbabilityDensityFunction = None):
        self.f = f
        self.h = h

        self.dynamics_noise_pdf = dynamics_noise_pdf
        self.measurement_noise_pdf = measurement_noise_pdf
        self.x0_pdf = x0_pdf

        self.particles = self.x0_pdf.sample(N_particles)

        self.N_particles = N_particles
        self.x_hat_plus = self.particles.mean(axis=0)
        self.P_plus = self.x0_pdf.covariance_matrix
        

    def predict(self):
        """
        Propagates particles using the process model.

        Parameters:
            u (np.ndarray): Control input.
        """
        noise = self.dynamics_noise_pdf.sample(n_points=self.N_particles)
        x_minus = np.array([self.f(x, w) for x, w in zip(self.particles, noise)])
        return x_minus

    def update(self, y):
        """
        Updates particle weights based on measurement likelihood.

        Parameters:
            y (np.ndarray): Current measurement.
        """
        likelihoods = np.array([self.measurement_likelihood(p, y) for p in self.particles])
        self.weights *= likelihoods
        self.weights += 1.e-300      # Avoid division by zero
        self.weights /= np.sum(self.weights)

    def measurement_likelihood(self, x, y):
        """
        Computes the likelihood of a measurement given a particle state.

        Parameters:
            x (np.ndarray): Particle state.
            y (np.ndarray): Measurement.
        """
        measurement_prediction = self.h(x)
        cov = self.measurement_noise_pdf.covariance_matrix
        error = y - measurement_prediction
        return np.exp(-0.5 * error.T @ np.linalg.inv(cov) @ error) / np.sqrt(np.linalg.det(2 * np.pi * cov))
    
    def get_likelihoods(self, x_minus, y):
        likelihoods = np.array([self.measurement_likelihood(x, y)] for x in x_minus)
        return likelihoods

    def resample(self):
        """
        Resamples particles based on weights using systematic resampling.
        """
        positions = (np.arange(self.N) + np.random.random()) / self.N
        indexes = np.zeros(self.N, 'i')
        cumulative_sum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < self.N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        self.particles = self.particles[indexes]
        self.weights.fill(1.0 / self.N)

    def estimate(self):
        """
        Computes the weighted mean and covariance of the particles.

        Returns:
            np.ndarray: Mean state estimate.
            np.ndarray: Covariance of the estimate.
        """
        mean = np.average(self.particles, weights=self.weights, axis=0)
        covariance = np.cov(self.particles.T, aweights=self.weights)
        return mean, covariance