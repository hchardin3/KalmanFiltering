"""
The unscented kalman filter.
"""

import numpy as np
from scipy.linalg import cholesky

class UnscentedKalmanFilter:
    def __init__(self, f, h, Q: np.ndarray, R: np.ndarray, x0: np.ndarray, P0: np.ndarray, kappa: float = 0.):
        """
        Initializes the Unscented Kalman Filter (UKF), an advanced filter that utilizes a deterministic 
        sampling technique to capture the mean and covariance estimates with higher accuracy than the 
        Extended Kalman Filter, especially for non-linear models.

        The UKF uses sigma points to approximate the probability distribution of the state. These points 
        are propagated through the non-linear functions, from which the new mean and covariance estimates 
        are derived.

        Parameters:
            f (callable): State transition function, which models the next state from the current state,
                          control input, and time. It should take the form f(x, u, t) -> np.ndarray.
            h (callable): Measurement function, which predicts the measurement expected from the current state.
                          It should take the form h(x, t) -> np.ndarray.
            Q (np.ndarray): Process noise covariance matrix, which models the uncertainty in the process or model.
            R (np.ndarray): Measurement noise covariance matrix, which models the uncertainty in sensor measurements.
            x0 (np.ndarray): Initial state estimate, a vector representing the estimated state of the system at t0.
            P0 (np.ndarray): Initial covariance estimate, a matrix representing the estimated accuracy of the state estimate.
            kappa (float): Scaling parameter for the sigma points, which affects the spread of the sigma points around
                           the mean state estimate. Typically, kappa is set to 0 for optimal performance in most applications.

        Example:
            # Define state transition and measurement functions
            def transition_function(x, u, t):
                return np.array([x[0] + u[0]*t + np.random.normal(), x[1] + u[1]*t + np.random.normal()])

            def measurement_function(x, t):
                return np.array([x[0] + np.random.normal(), x[1] + np.random.normal()])

            # Initialize the filter
            ukf = UnscentedKalmanFilter(transition_function, measurement_function,
                                        Q=np.diag([0.1, 0.1]), R=np.diag([0.1, 0.1]),
                                        x0=np.array([0, 0]), P0=np.eye(2), kappa=0)
        """
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R
        self.P_plus = P0  # Estimation error covariance
        self.x_hat_plus = x0  # State estimate
        self.kappa = kappa  # Scaling parameter for the sigma points
        self.n = x0.size  # Dimension of the state
        self.Kalman = None

    def generate_sigma_points(self, x_hat = None, P = None):
        """
        Generates sigma points for the current state estimate and covariance matrix.

        Returns:
            np.ndarray: Sigma points array.
        """
        n = self.n

        if x_hat is None:
            x_hat = self.x_hat_plus

        if P is None:
            P = self.P_plus

        # Compute the square root of (n + kappa) * P
        S = cholesky((n + self.kappa) * P, lower=True)
        sigma_points = np.zeros((2 * n , n))

        for i in range(n):
            sigma_points[i] = x_hat + S[:, i]
            sigma_points[n + i] = x_hat - S[:, i]

        return sigma_points

    def predict(self, sigma_points: np.ndarray, u: np.ndarray, t: int|float):
        """
        Predicts the state estimate and covariance matrix using sigma points.

        Parameters:
            sigma_points (np.ndarray): Sigma points. 
            u (np.ndarray): Control input vector.
            t (int or float): Time.
        """
        X_minus = np.array([self.f(x, u, t) for x in sigma_points])  # Apply the state transition function

        # Calculate predicted state mean
        x_hat_minus = np.mean(X_minus, axis=0)

        # Calculate predicted covariance
        P_minus = np.sum([(x - x_hat_minus)[:, np.newaxis] @ (x - x_hat_minus)[np.newaxis, :] for x in X_minus], axis=0) / (2 * self.n) + self.Q

        return x_hat_minus, P_minus

    def update(self, y: np.ndarray, u: np.ndarray, t: int|float, Qk: np.ndarray = None, Rk: np.ndarray = None, best_precision: bool = True):
        """
        Updates the state estimate and covariance based on the measurement.

        Parameters:
            y (np.ndarray): Measurement vector.
            u (np.ndarray): Control input vector.
            t (int or float): Time.
            Qk (np.ndarray, optional): Updated value of process noise if it changes over time.
            Rk (np.ndarray, optional): Updated value of measurement noise if it changes over time.
            best_precision (boolean, True by default): Wether we recompute sigma points for y (better precision but consumes more resources).
        """
        sigma_points = self.generate_sigma_points()
        x_hat_minus, P_minus = self.predict(sigma_points, u, t)

        # Update the covariance matrixes if necessary:
        if Qk is not None:
            self.Q = Qk
        if Rk is not None:
            self.R = Rk

        if best_precision:
            sigma_points = self.generate_sigma_points(x_hat_minus, P_minus)
        
        Y = np.array([self.h(x, t) for x in sigma_points])  # Apply the measurement function

        # Calculate measurement prediction
        y_hat = np.mean(Y, axis=0)

        # Calculate innovation covariance matrix
        P_yy = np.sum([(y - y_hat)[:, np.newaxis] @ (y - y_hat)[np.newaxis, :] for y in Y], axis=0) / (2 * self.n) + self.R
        P_xy = np.sum([(sigma_points[i] - x_hat_minus)[:, np.newaxis] @ (Y[i] - y_hat)[np.newaxis, :] for i in range(2*self.n)], axis=0) / (2 * self.n)

        # Calculate Kalman gain
        self.Kalman = P_xy @ np.linalg.inv(P_yy)

        # Update the state estimate
        self.x_hat_plus = x_hat_minus + self.Kalman @ (y - y_hat)

        # Update the covariance matrix
        if P_yy.shape[0] == 1:
            self.P_plus = P_minus - self.Kalman  @ self.Kalman.T * P_yy
        else:
            self.P_plus = P_minus - self.Kalman @ P_yy @ self.Kalman.T

    def get_estimate(self):
        """
        Returns the current state estimate.

        Returns:
            np.ndarray: The current state estimate.
        """
        return self.x_hat_plus

    def get_error_covariance(self):
        """
        Returns the current estimation error covariance.

        Returns:
            np.ndarray: The current estimation error covariance.
        """
        return self.P_plus
