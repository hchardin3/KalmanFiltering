import numpy as np
import math
from numpy.linalg import det, inv, slogdet, cholesky
from scipy.stats import multivariate_normal

class ProbabilityDensityFunction:
    def __init__(self, pdf, n_dim: int = 1, mean: np.ndarray = None, covariance_matrix: np.ndarray = None) -> None:
        """
        A Probability Density Function is assumed to cover the vector space R^n_dim, and to have the regular properties of the pdf.

        Parameters:
            pdf (callable): A probability density function over R^n_dim. Takes a vector as an input and returns a probability. Is assumed to always be positive, and has an integral of 1.
            n_dim (int): The dimension of the space.
            mean (np.ndarray): The mean of the distribution, if known.
            covariance_matrix (np.ndarray): The covariance matrix of the distribution, if known.
        """
        self.pdf = pdf
        self.n_dim = n_dim
        self.mean = mean
        self.covariance_matrix = covariance_matrix
        self.space_bounds = None

        if self.mean is None or self.covariance_matrix is None:
            samples = self.sample(10000)  # Use a large number of samples to estimate
            if self.mean is None:
                self.mean = self.compute_mean(samples)
            if self.covariance_matrix is None:
                self.covariance_matrix = self.compute_covariance(samples)
        
        self.space_bounds = self.compute_space_bounds()

    def evaluate(self, x):
        return self.pdf(x)
    
    def sample(self, n_points, space_bounds=None):
        """
        Sample N random points according to the probability density function.
        
        Parameters:
        n_points: int - Number of points to sample
        space_bounds: list of tuples - Bounds for each dimension [(min1, max1), (min2, max2), ...]
        
        Returns:
        samples: np.ndarray - Sampled points
        """
        if self.mean is not None and self.covariance_matrix is not None:
            # If mean and covariance are provided, use multivariate normal sampling
            mvn = multivariate_normal(mean=self.mean, cov=self.covariance_matrix)
            samples = mvn.rvs(size=n_points)
        else:
            if space_bounds is None:
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
    
    def compute_space_bounds(self, bound_factor: int | float = 10):
        """
        Compute the space bounds based on the mean and covariance matrix.

        Parameters:
            bound_factor (int or float): A number to scale the bound.
        
        Returns:
        space_bounds: list of tuples - Computed bounds for each dimension
        """
        max_eigenvalue = np.max(np.linalg.eigvals(self.covariance_matrix))
        x = np.abs(self.mean) * bound_factor * max_eigenvalue
        space_bounds = [(self.mean[i] - x[i], self.mean[i] + x[i]) for i in range(self.n_dim)]
        return space_bounds

class GaussianPDF(ProbabilityDensityFunction):
    def __init__(self, mean: np.ndarray|None = None, covariance_matrix: np.ndarray|None = None) -> None:
        if mean is None:
            mean = np.zeros(2)  # Default to 2D for simplicity if not specified
        if covariance_matrix is None:
            covariance_matrix = np.eye(len(mean))  # Identity matrix of dimension len(mean)

        n_dim = mean.size

        # Define the pdf function using a closure to capture mean and covariance_matrix
        def pdf(x: np.ndarray):
            # Calculate the normalizing constant
            denom = np.sqrt((2 * np.pi) ** self.n_dim * det(covariance_matrix))
            # Calculate the exponent term
            x_m = x - mean
            exponent = -0.5 * np.dot(x_m.T, np.dot(inv(covariance_matrix), x_m))
            return np.exp(exponent) / denom

        super().__init__(pdf, n_dim, mean, covariance_matrix)
    
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


class ParticleFilter:
    def __init__(self, f, h, N_particles: int, dynamics_noise_pdf: ProbabilityDensityFunction, measurement_noise_pdf: ProbabilityDensityFunction, x0_pdf: ProbabilityDensityFunction | None = None, x0: np.ndarray | None = None, P0: np.ndarray | None = None):
        """
        Initialization of a classic particle filter. 

        Parameters:
            f (callable): State transition function. Should take arguments (x, w) and return the new state.
            h (callable): Measurement function. Should take arguments (x, v) and return the measurement.
            N_particles (int): Number of particle in the filter. A higher number means enhanced precision but heavier computation on each steps.
            dynamics_noise_pdf (ProbabilityDensityFunction): The PDF of w the dynamics noise.
            measurement_noise_pdf (ProbabilityDensityFunction): The PDF of v the measurement noise.
            x0_pdf (ProbabilityDensityFunction): The PDF of the initial state x0. If None, then x0 and P0 have to be initialized (the PDF will then be assumed to be a gaussian).
            x0 (np.ndarray): Initial state estimate vector, if known. Optional if x0_pdf is inputed.
            P0 (np.ndarray): Initial estimation error covariance matrix, if known. Optional if x0_pdf is inputed.
        """
        if x0_pdf is None and (x0 is None or P0 is None):
            raise(TypeError("Either specify the PDF of x0 or its mean and distribution covariance"))
        
        self.f = f
        self.h = h

        self.dynamics_noise_pdf = dynamics_noise_pdf
        self.measurement_noise_pdf = measurement_noise_pdf
        self.x0_pdf = x0_pdf

        if x0_pdf is None:
            self.x0_pdf = GaussianPDF(x0, P0)

        self.particles = self.x0_pdf.sample(N_particles)
        self.weights = np.ones(N_particles) / N_particles   

        self.N_particles = N_particles
        self.x_hat_plus = self.particles.mean(axis=0)
        self.P_plus = np.cov(self.particles.T, aweights=self.weights)

        self.system_size = self.x_hat_plus.size

        self.R_inv = np.linalg.inv(self.measurement_noise_pdf.covariance_matrix)

       

    def predict(self):
        """
        Propagates particles using the process model.
        """
        noise = self.dynamics_noise_pdf.sample(n_points=self.N_particles)
        x_minus = np.array([self.f(x, w) for x, w in zip(self.particles, noise)])
        return x_minus

    def update(self, y, regularized_resampling: bool = True, fk = None, hk = None, dynamics_noise_pdf_k = None, measurement_noise_pdf_k = None):
        """
        Updates particle weights based on measurement likelihood.

        Parameters:
            y (np.ndarray): Current measurement.
            regularized_resampling (bool): Wether to use the Regularized Resampling method or not. This method is more precise but heavier. True by default.
        """
        x_minus = self.predict()

        if fk is not None:
            self.f = fk
        if hk is not None:
            self.h = hk
        if dynamics_noise_pdf_k is not None:
            self.dynamics_noise_pdf = dynamics_noise_pdf_k
        if measurement_noise_pdf_k is not None:
            self.measurement_noise_pdf = measurement_noise_pdf_k

        likelihoods = np.array([self.measurement_likelihood(x, y) for x in x_minus])

        # Update weights
        self.weights = likelihoods
        self.weights += 1.e-300      # Avoid division by zero
        self.weights /= np.sum(self.weights)

        # Resample particles based on updated weights
        if regularized_resampling:
            self.regularized_resampling(x_minus)
        else:
            self.resample()
        
        # Update mean and covariance estimates
        self.estimate()

    def measurement_likelihood(self, x, y):
        """
        Computes the likelihood of a measurement given a particle state.

        Parameters:
            x (np.ndarray): Particle state.
            y (np.ndarray): Measurement.
        """
        measurement_prediction = self.h(x)            
        error = y - measurement_prediction

        return np.exp(-0.5 * error.T @ self.R_inv @ error) / np.sqrt(np.linalg.det(2 * np.pi * self.measurement_noise_pdf.covariance_matrix))

    def estimate(self):
        """
        Computes the weighted mean and covariance of the particles.

        Returns:
            np.ndarray: Mean state estimate.
            np.ndarray: Covariance of the estimate.
        """
        self.x_hat_plus = np.average(self.particles, weights=self.weights, axis=0)
        self.P_plus = np.cov(self.particles.T, aweights=self.weights)
    
    def resample(self):
        """
        Resamples particles based on weights using systematic resampling.
        """
        positions = (np.arange(self.N_particles) + np.random.random()) / self.N_particles
        indexes = np.zeros(self.N_particles, 'i')
        cumulative_sum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < self.N_particles:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        self.particles = self.particles[indexes]
        self.weights.fill(1.0 / self.N_particles)
    
    def regularized_resampling(self, x_minus: np.ndarray):
        """
        The regularized resampling method to get resampled particles from the raw particles x_minus.

        Parameters:
            x_minus (np.array): The raw particles that went through the dynamics function f.
        """
        S = np.cov(x_minus.T)

        A = cholesky(S)

        if self.system_size == 1:
            v = 2
        elif self.system_size == 2:
            v = math.pi
        else:
            v = 2 * (math.pi ** (self.system_size / 2)) / math.gamma(self.system_size / 2)

        h = 0.5 * ((8/v) * (self.system_size + 4) * ((2 * math.sqrt(math.pi))**self.system_size)) ** (1/(self.system_size+4)) * (self.N_particles ** (-1/(self.system_size+4)))

        # Compute the kernel density estimation
        K_h = lambda x: (1 / (det(A) * h ** self.system_size)) * self.epanechnikov_kernel(inv(A) @ x / h, v)

        # Approximate the PDF p(x_k | y_k)
        p_x_given_y = np.zeros(self.N_particles)
        for i in range(self.N_particles):
            p_x_given_y[i] = np.sum([self.weights[j] * K_h(x_minus[i] - x_minus[j]) for j in range(self.N_particles)])

        # Normalize the PDF
        p_x_given_y /= np.sum(p_x_given_y)

        # Resample particles based on the approximated PDF
        indexes = np.random.choice(np.arange(self.N_particles), size=self.N_particles, p=p_x_given_y)
        self.particles = x_minus[indexes]

    def epanechnikov_kernel(self, x: np.array, v: float):
        """
        Epanechnikov kernel function.

        Parameters:
            x (np.ndarray): Input vector.
            v (float): Additional parameter.

        Returns:
            float: Kernel value.
        """
        norm_x = np.linalg.norm(x)
        if norm_x < 1:
            return (1 - norm_x ** 2) * ((self.system_size + 2) / (2 * v))
        else:
            return 0.0


class ExtendedParticleFilter:
    def __init__(self, f, F_jac, h, H_jac, N_particles: int, dynamics_noise_pdf: ProbabilityDensityFunction, measurement_noise_pdf: ProbabilityDensityFunction, x0_pdf: ProbabilityDensityFunction | None = None, x0: np.ndarray | None = None, P0: np.ndarray | None = None):
        """
        Initialization of an extended Kalman particle filter. 

        Parameters:
            f (callable): State transition function. Should take arguments (x, w) and return the new state.
            F_jac (callable): Space jacobian of f.
            h (callable): Measurement function. Should take arguments (x, v) and return the measurement.
            H_jac (callable): Space jacobian of h.
            N_particles (int): Number of particle in the filter. A higher number means enhanced precision but heavier computation on each steps.
            dynamics_noise_pdf (ProbabilityDensityFunction): The PDF of w the dynamics noise.
            measurement_noise_pdf (ProbabilityDensityFunction): The PDF of v the measurement noise.
            x0_pdf (ProbabilityDensityFunction): The PDF of the initial state x0. If None, then x0 and P0 have to be initialized (the PDF will then be assumed to be a gaussian).
            x0 (np.ndarray): Initial state estimate vector, if known. Optional if x0_pdf is inputed.
            P0 (np.ndarray): Initial estimation error covariance matrix, if known. Optional if x0_pdf is inputed.
        """
        
        
        self.f = f
        self.F_jac = F_jac
        self.h = h
        self.H_jac = H_jac

        self.dynamics_noise_pdf = dynamics_noise_pdf
        self.measurement_noise_pdf = measurement_noise_pdf
        self.x0_pdf = x0_pdf

        if x0_pdf is None:
            self.x0_pdf = GaussianPDF(x0, P0)

        self.particles = self.x0_pdf.sample(N_particles)
        self.weights = np.ones(N_particles) / N_particles

        self.N_particles = N_particles
        self.x_hat_plus = self.particles.mean(axis=0)
        self.P_plus_0 = np.cov(self.particles.T, aweights=self.weights)
        self.P_plus = [self.P_plus_0 for _ in range(self.N_particles)]

        self.system_size = self.x_hat_plus.size

        self.Q = dynamics_noise_pdf.covariance_matrix
        self.R = measurement_noise_pdf.covariance_matrix

        self.R_inv = np.linalg.inv(self.R)

    
    def predict(self):
        """
        Propagates particles using the process model.
        """
        noise = self.dynamics_noise_pdf.sample(n_points=self.N_particles)
        x_minus = np.array([self.f(x, w) for x, w in zip(self.particles, noise)])

        P_minus = [self.F_jac(self.particles[k]) @ self.P_plus[k] @ self.F_jac(self.particles[k]).T + self.Q for k in range(self.N_particles)]

        return x_minus, P_minus
    
    def update(self, y, regularized_resampling: bool = True, fk=None, F_jack = None, hk=None, H_jack = None, dynamics_noise_pdf_k=None, measurement_noise_pdf_k=None):
        """
        Updates particle weights based on measurement likelihood.

        Parameters:
            y (np.ndarray): Current measurement.
            regularized_resampling (bool): Whether to use the Regularized Resampling method or not. This method is more precise but heavier. True by default.
        """
        x_minus, P_minus = self.predict()

        if fk is not None:
            self.f = fk
        if F_jack is not None:
            self.F_jac = F_jack
        if hk is not None:
            self.h = hk
        if H_jack is not None:
            self.H_jac = H_jack
        if dynamics_noise_pdf_k is not None:
            self.dynamics_noise_pdf = dynamics_noise_pdf_k
        if measurement_noise_pdf_k is not None:
            self.measurement_noise_pdf = measurement_noise_pdf_k

        H = [self.H_jac(self.particles[k]) for k in range(self.N_particles)]
        S = [H[k] @ P_minus[k] @ H[k].T + self.R for k in range(self.N_particles)]
        self.Kalman = [P_minus[k] @ H[k].T @ inv(S[k]) for k in range(self.N_particles)]

        self.particles = np.array([x_minus[k] + self.Kalman[k].dot(y - self.h(self.particles[k])) for k in range(self.N_particles)])
        self.P_plus = [(np.eye(self.system_size) - self.Kalman[k] @ H[k]) @ P_minus[k] for k in range(self.N_particles)]

        # Update weights based on measurement likelihood
        likelihoods = np.array([self.measurement_likelihood(x, y) for x in self.particles])

        # Update weights
        self.weights = likelihoods
        self.weights += 1.e-300      # Avoid division by zero
        self.weights /= np.sum(self.weights)

        # Resample particles based on updated weights
        if regularized_resampling:
            self.regularized_resampling()
        else:
            self.resample()
        
        # Update mean and covariance estimates
        self.estimate()
    
    def measurement_likelihood(self, x, y):
        """
        Computes the likelihood of a measurement given a particle state.

        Parameters:
            x (np.ndarray): Particle state.
            y (np.ndarray): Measurement.
        """
        measurement_prediction = self.h(x)
        error = y - measurement_prediction
        return np.exp(-0.5 * error.T @ self.R_inv @ error) / np.sqrt(np.linalg.det(2 * np.pi * self.R))

    def estimate(self):
        """
        Computes the weighted mean and covariance of the particles.

        Returns:
            np.ndarray: Mean state estimate.
            np.ndarray: Covariance of the estimate.
        """
        mean = np.average(self.particles, weights=self.weights, axis=0)
        covariance = np.cov(self.particles.T, aweights=self.weights)
        self.x_hat_plus = mean
        self.P_plus_0 = covariance

    def resample(self):
        """
        Resamples particles based on weights using systematic resampling.
        """
        positions = (np.arange(self.N_particles) + np.random.random()) / self.N_particles
        indexes = np.zeros(self.N_particles, 'i')
        cumulative_sum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < self.N_particles:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        self.particles = self.particles[indexes]
        self.P_plus = self.P_plus[indexes]
        self.weights.fill(1.0 / self.N_particles)
    
    def regularized_resampling(self):
        """
        The regularized resampling method to get resampled particles from the raw particles x_minus.

        Parameters:
            x_minus (np.array): The raw particles that went through the dynamics function f.
        """
        S = np.cov(self.particles.T)

        A = cholesky(S)

        if self.system_size == 1:
            v = 2
        elif self.system_size == 2:
            v = math.pi
        else:
            v = 2 * (math.pi ** (self.system_size / 2)) / math.gamma(self.system_size / 2)

        h = 0.5 * ((8/v) * (self.system_size + 4) * ((2 * math.sqrt(math.pi))**self.system_size)) ** (1/(self.system_size+4)) * (self.N_particles ** (-1/(self.system_size+4)))

        # Compute the kernel density estimation
        K_h = lambda x: (1 / (det(A) * h ** self.system_size)) * self.epanechnikov_kernel(inv(A) @ x / h, v)

        # Approximate the PDF p(x_k | y_k)
        p_x_given_y = np.zeros(self.N_particles)
        for i in range(self.N_particles):
            p_x_given_y[i] = np.sum([self.weights[j] * K_h(self.particles[i] - self.particles[j]) for j in range(self.N_particles)])

        # Normalize the PDF
        p_x_given_y /= np.sum(p_x_given_y)

        # Resample particles based on the approximated PDF
        indexes = np.random.choice(np.arange(self.N_particles), size=self.N_particles, p=p_x_given_y)
        self.particles = self.particles[indexes]
        self.P_plus = self.P_plus[indexes]
        self.weights.fill(1.0 / self.N_particles)

    def epanechnikov_kernel(self, x: np.array, v: float):
        """
        Epanechnikov kernel function.

        Parameters:
            x (np.ndarray): Input vector.
            v (float): Additional parameter.

        Returns:
            float: Kernel value.
        """
        norm_x = np.linalg.norm(x)
        if norm_x < 1:
            return (1 - norm_x ** 2) * ((self.system_size + 2) / (2 * v))
        else:
            return 0.0