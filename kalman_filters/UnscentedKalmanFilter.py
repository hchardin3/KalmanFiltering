import numpy as np
from scipy.linalg import cholesky

class UnscentedKalmanFilter:
    def __init__(self, f, h, Q: np.ndarray, R: np.ndarray, P0: np.ndarray, x0: np.ndarray, kappa: float = 0.):
        """
        Initializes the Unscented Kalman Filter with the provided functions and matrices.

        Parameters:
            f (callable): State transition function. Should take arguments (x, u, t) and return the new state.
            h (callable): Measurement function. Should take arguments (x, t) and return the measurement.
            Q (np.ndarray): Process noise covariance matrix.
            R (np.ndarray): Measurement noise covariance matrix.
            P0 (np.ndarray): Initial estimation error covariance matrix.
            x0 (np.ndarray): Initial state estimate vector.
            kappa (float): Scaling parameter for the sigma points.
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
        self.P_plus = P_minus - self.Kalman @ P_yy @ self.Kalman.T

    def get_estimate(self):
        """
        Returns the current state estimate.

        Returns:
            np.ndarray: The current state estimate.
        """
        return self.x_hat_plus

    def get_precision(self):
        """
        Returns the current estimation error covariance.

        Returns:
            np.ndarray: The current estimation error covariance.
        """
        return self.P_plus
