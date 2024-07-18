"""
This script defines two types of filters:
    ParticleFilter: The classic particle filter.
    ExtendedParticleFilter: A particle filter that uses equations of extended kalman filter in its update.
"""

import numpy as np
import math
from numpy.linalg import det, inv, cholesky
from scipy.stats import multivariate_normal
from stats.probability_density import ProbabilityDensityFunction
from stats.multivariate_pdf import MultivariateGaussianPDF
from stats.univariate_pdf import UnivariateGaussianPDF

class ParticleFilter:
    def __init__(self, f, h, N_particles: int, dynamics_noise_pdf: ProbabilityDensityFunction, measurement_noise_pdf: ProbabilityDensityFunction, x0_pdf: ProbabilityDensityFunction | None = None, x0: np.ndarray | None = None, P0: np.ndarray | None = None):
        """
        Initializes a classic Particle Filter, which uses a set of particles to represent the posterior distributions
        of the state space. This method is particularly effective for non-linear and non-Gaussian state estimation problems.

        Parameters:
            f (callable): State transition function, defines how the state evolves. It should take the current state x and process noise w, returning the next state.
            h (callable): Measurement function, defines how the measurements are generated from the state. It should take the current state x and measurement noise v, returning the measurement.
            N_particles (int): Number of particles to use in the filter. Increasing the number of particles can improve  the approximation at the cost of computational complexity.
            dynamics_noise_pdf (ProbabilityDensityFunction): Probability density function for the process noise w.
            measurement_noise_pdf (ProbabilityDensityFunction): Probability density function for the measurement noise v.
            x0_pdf (ProbabilityDensityFunction, optional): Probability density function for generating the initial state particles. If not provided, x0 and P0 must be given to define  a Gaussian distribution.
            x0 (np.ndarray, optional): Initial state estimate vector, used if x0_pdf is not provided.
            P0 (np.ndarray, optional): Initial covariance matrix, used if x0_pdf is not provided.

        Raises:
            ValueError: If neither x0_pdf is provided nor both x0 and P0 are specified.
        """
        if x0_pdf is None and (x0 is None or P0 is None):
            raise(ValueError("User should specify either x0_pdf or both x0 and P0"))
        
        self.f = f
        self.h = h

        self.dynamics_noise_pdf = dynamics_noise_pdf
        self.measurement_noise_pdf = measurement_noise_pdf

        if x0_pdf is None:
            if x0.size == 1:
                self.x0_pdf = UnivariateGaussianPDF(x0, P0)
            else:
                self.x0_pdf = MultivariateGaussianPDF(x0, P0)
        else:
            self.x0_pdf = x0_pdf

        self.particles = self.x0_pdf.sample(N_particles)
        self.weights = np.ones(N_particles) / N_particles   

        self.N_particles = N_particles
        self.x_hat_plus = self.particles.mean(axis=0)
        self.P_plus = np.cov(self.particles.T, aweights=self.weights)

        self.system_size = self.x_hat_plus.size

        self.R_inv = np.linalg.inv(self.measurement_noise_pdf.covariance_matrix)

        self.privilege_precision = True
        self.num_likelihood_samples = 10

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
        Computes the likelihood of a measurement given a particle state using the multivariate normal distribution.

        Parameters:
            x (np.ndarray): Particle state.
            y (np.ndarray): Measurement.
        """
        if self.privilege_precision:
            noise_samples = self.measurement_noise_pdf.sample(self.num_likelihood_samples)
            
            # Pre-compute the multivariate normal distribution for efficiency
            mvn = multivariate_normal(mean=np.zeros(self.measurement_noise_pdf.n_dim), cov=self.measurement_noise_pdf.covariance_matrix)
            
            probabilities = []
            
            for v in noise_samples:
                predicted_measurement = self.h(x, v)
                error = y - predicted_measurement
                # Evaluate the probability density function for the given error
                prob = mvn.pdf(error)
                probabilities.append(prob)
            
            # Return the average probability as the likelihood
            return np.mean(probabilities)
        else:
            mvn = multivariate_normal(mean=self.measurement_noise_pdf.mean, cov=self.measurement_noise_pdf.covariance_matrix)
            predicted_measurement = self.h(x, self.measurement_noise_pdf.mean)
            error = y - predicted_measurement

            return mvn.pdf(error)


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

        if self.system_size == 1:
            A = np.sqrt(S)
            v = 2
        elif self.system_size == 2:
            v = math.pi
            A = cholesky(S)
        else:
            A = cholesky(S)
            v = 2 * (math.pi ** (self.system_size / 2)) / math.gamma(self.system_size / 2)

        h = 0.5 * ((8/v) * (self.system_size + 4) * ((2 * math.sqrt(math.pi))**self.system_size)) ** (1/(self.system_size+4)) * (self.N_particles ** (-1/(self.system_size+4)))

        # Compute the kernel density estimation
        if self.system_size == 1:
            K_h = lambda x: (1 / (A * h)) * self.epanechnikov_kernel(x / (A * h), v)
        else:
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
    
    def get_estimate(self):
        """
        Returns the current state estimate.
        
        Returns:
            np.ndarray: The current state estimate.
        """
        return self.x_hat_plus


class ExtendedParticleFilter:
    def __init__(self, f, F_jac, h, H_jac, N_particles: int, dynamics_noise_pdf: ProbabilityDensityFunction, measurement_noise_pdf: ProbabilityDensityFunction, x0_pdf: ProbabilityDensityFunction | None = None, x0: np.ndarray | None = None, P0: np.ndarray | None = None):
        """
        Initializes an Extended Particle Filter, which integrates the linearized approach of the Extended Kalman Filter
        within the particle filter framework. This filter is useful for systems where non-linearities can be approximately
        linearized around the current estimate but still benefit from a particle-based representation.

        Parameters:
            f (callable): Non-linear state transition function, similar to the basic Particle Filter.
            F_jac (callable): Jacobian of the state transition function f, used to linearize the process model.
            h (callable): Non-linear measurement function, similar to the basic Particle Filter.
            H_jac (callable): Jacobian of the measurement function h, used to linearize the measurement model.
            N_particles (int): Number of particles to use in the filter.
            dynamics_noise_pdf (ProbabilityDensityFunction): PDF for the process noise.
            measurement_noise_pdf (ProbabilityDensityFunction): PDF for the measurement noise.
            x0_pdf (ProbabilityDensityFunction, optional): PDF for generating the initial state particles.
            x0 (np.ndarray, optional): Initial state estimate vector, used if x0_pdf is not provided.
            P0 (np.ndarray, optional): Initial covariance matrix, used if x0_pdf is not provided.

            
        Raises:
            ValueError: If neither x0_pdf is provided nor both x0 and P0 are specified.
        """
        
        if x0_pdf is None and (x0 is None or P0 is None):
            raise(ValueError("User should specify either x0_pdf or both x0 and P0"))
        
        self.f = f
        self.F_jac = F_jac
        self.h = h
        self.H_jac = H_jac

        self.dynamics_noise_pdf = dynamics_noise_pdf
        self.measurement_noise_pdf = measurement_noise_pdf

        if x0_pdf is None:
            self.x0_pdf = MultivariateGaussianPDF(x0, P0)
        else:
            self.x0_pdf = x0_pdf

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

        self.privilege_precision = True
        self.num_likelihood_samples = 10

    
    def predict(self):
        """
        Propagates particles using the process model.
        """
        noise = self.dynamics_noise_pdf.sample(n_points=self.N_particles)
        x_minus = np.array([self.f(x, w) for x, w in zip(self.particles, noise)])

        if self.x_hat_plus.size == 1:
            P_minus = [self.F_jac(self.particles[k]) * self.P_plus[k] * self.F_jac(self.particles[k]).T + self.Q for k in range(self.N_particles)]
        else:
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

        if self.R.shape[0] == 1:            
            self.Kalman = [(P_minus[k] @ H[k].T * inv(S[k])).reshape(self.system_size) for k in range(self.N_particles)]
            self.particles = np.array([x_minus[k] + (self.Kalman[k].T * (y - self.h(self.particles[k], self.measurement_noise_pdf.mean))) for k in range(self.N_particles)])
        else:
            self.Kalman = [P_minus[k] @ H[k].T @ inv(S[k]) for k in range(self.N_particles)]
            self.particles = np.array([x_minus[k] + (self.Kalman[k].dot(y - self.h(self.particles[k], self.measurement_noise_pdf.mean))) for k in range(self.N_particles)])
        
        
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
        Computes the likelihood of a measurement given a particle state using the multivariate normal distribution.

        Parameters:
            x (np.ndarray): Particle state.
            y (np.ndarray): Measurement.
        """
        if self.privilege_precision:
            noise_samples = self.measurement_noise_pdf.sample(self.num_likelihood_samples)
            
            # Pre-compute the multivariate normal distribution for efficiency
            mvn = multivariate_normal(mean=np.zeros(self.measurement_noise_pdf.n_dim), cov=self.measurement_noise_pdf.covariance_matrix)
            
            probabilities = []
            
            for v in noise_samples:
                predicted_measurement = self.h(x, v)
                error = y - predicted_measurement
                # Evaluate the probability density function for the given error
                prob = mvn.pdf(error)
                probabilities.append(prob)
            
            # Return the average probability as the likelihood
            return np.mean(probabilities)
        else:
            mvn = multivariate_normal(mean=self.measurement_noise_pdf.mean, cov=self.measurement_noise_pdf.covariance_matrix)
            predicted_measurement = self.h(x, self.measurement_noise_pdf.mean)
            error = y - predicted_measurement

            return mvn.pdf(error)

    def estimate(self):
        """
        Computes the weighted mean and covariance of the particles.

        Returns:
            np.ndarray: Mean state estimate.
            np.ndarray: Covariance of the estimate.
        """
        self.x_hat_plus = np.average(self.particles, weights=self.weights, axis=0)
        self.P_plus_0 = np.cov(self.particles.T, aweights=self.weights)

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

        if self.system_size == 1:
            A = np.sqrt(S)
            v = 2
        elif self.system_size == 2:
            A = cholesky(S)
            v = math.pi
        else:
            A = cholesky(S)
            v = 2 * (math.pi ** (self.system_size / 2)) / math.gamma(self.system_size / 2)

        h = 0.5 * ((8/v) * (self.system_size + 4) * ((2 * math.sqrt(math.pi))**self.system_size)) ** (1/(self.system_size+4)) * (self.N_particles ** (-1/(self.system_size+4)))

        # Compute the kernel density estimation
        if self.system_size == 1:
            K_h = lambda x: (1 / (A * h)) * self.epanechnikov_kernel(x / (h * A), v)
        else:
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

        # Convert self.P_plus to a numpy array before indexing
        self.P_plus = np.array(self.P_plus)[indexes]
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
    
    def get_estimate(self):
        """
        Returns the current state estimate.
        
        Returns:
            np.ndarray: The current state estimate.
        """
        return self.x_hat_plus