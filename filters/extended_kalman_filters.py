"""
This script defines two types of filters:
    HybridExtendedKalmanFilter: An extended kalman filter that works for systems with continuous dynamics and discrete measurements at a regular time.
    DiscreteExtendedKalmanFilter: An extended kalman filter that works for systems with discrete dynamics and discrete measurements.
"""

import numpy as np
from scipy.integrate import odeint

class HybridExtendedKalmanFilter:
    def __init__(self, f, h, F_jacobian, H_jacobian, L, M, Q: np.ndarray, R: np.ndarray, x0: np.ndarray, P0: np.ndarray, t_span: float|int, precision: int=100):
        """
        Initializes a Hybrid Extended Kalman Filter designed for systems with continuous dynamics and discrete measurements.
        This filter integrates state estimates and their uncertainties over time, updating them at discrete measurement points.

        Parameters:
            f (callable): Continuous-time state transition function modeling the dynamics, taking state x, control u, process noise w, and time t, returning the state derivative.
            h (callable): Measurement function modeling the observation process, taking state x, measurement noise v, and time t, returning the measured output.
            F_jacobian (callable): Function to compute the Jacobian matrix of f with respect to state x.
            H_jacobian (callable): Function to compute the Jacobian matrix of h with respect to state x.
            L (callable): Function to compute the matrix derivative of f with respect to process noise w.
            M (callable): Function to compute the matrix derivative of h with respect to measurement noise v.
            Q (np.ndarray): Process noise covariance matrix, quantifying the uncertainty in the process model.
            R (np.ndarray): Measurement noise covariance matrix, quantifying the uncertainty in the measurement process.
            x0 (np.ndarray): Initial state estimate vector.
            P0 (np.ndarray): Initial estimation error covariance matrix.
            t_span (float|int): Duration over which the continuous dynamics are integrated.
            precision (int): Number of points for numerical integration, balancing precision with computational load.

        The filter uses numerical integration to evolve state estimates between measurements, making it suitable for
        systems where an analytical solution to the state equations is not feasible.
        """
        self.f = f
        self.h = h
        self.F_jacobian = F_jacobian
        self.H_jacobian = H_jacobian
        self.L = L
        self.M = M
        self.Q = Q
        self.R = R
        self.P_plus = P0  # Initial estimation error covariance
        self.x_hat_plus = x0  # Initial state estimate
        self.t_span = t_span  # Time span for each integration step
        self.k = 0  # Current time step
        self.precision = precision  # Number of points for numerical integration

        self.Kalman = None

        self.xdim = x0.shape[0]  # Dimension of the state vector
    
    def combined_derivatives(self, combined: np.ndarray, t: float, u: np.ndarray):
        """
        Computes the derivatives of both the state estimate and the covariance.
        
        Parameters:
            combined (np.ndarray): Combined state estimate and covariance vector.
            t (float): Time.
            u (np.ndarray): Control input vector.
        
        Returns:
            np.ndarray: Derivatives of the combined state estimate and covariance.
        """
        n = self.x_hat_plus.shape[0]
        x_hat = combined[:n]  # Extract state estimate
        P_flat = combined[n:]  # Extract covariance
        P = P_flat.reshape(self.P_plus.shape)  # Reshape covariance to matrix
        
        # Compute the state estimate derivative
        x_hat_dot = self.f(x_hat, u, 0, t)
        
        # Compute the covariance derivative
        F = self.F_jacobian(x_hat, u, 0, t)
        L = self.L(x_hat, u, 0, t)
        P_dot = F.dot(P) + P.dot(F.T) + L.dot(self.Q).dot(L.T)
        
        # Combine the derivatives
        combined_dot = np.hstack((x_hat_dot, P_dot.flatten()))
        return combined_dot
    
    def predict(self, u: np.ndarray):
        """
        Integrates the state estimate and covariance from time (k-1)+ to time k-.
        
        Parameters:
            u (np.ndarray): Control input vector.
        """        
        # Combine the initial state estimate and covariance
        combined_initial = np.hstack((self.x_hat_plus, self.P_plus.flatten()))
        
        # Integrate the combined state estimate and covariance
        t_span = np.linspace(self.k * self.t_span, (self.k + 1) * self.t_span, self.precision)
        combined_result = odeint(self.combined_derivatives, combined_initial, t_span, args=(u,))[-1]
        
        # Extract the state estimate and covariance
        x_hat_minus = combined_result[:self.xdim]  # Extract state estimate
        P_flat = combined_result[self.xdim:]  # Extract covariance
        P_minus = P_flat.reshape(self.P_plus.shape)  # Reshape covariance to matrix

        return x_hat_minus, P_minus


    def update(self, u: np.ndarray, y: np.ndarray, hk = None, H_Jacobiank = None, Rk: np.ndarray = None):
        """
        Updates the state estimate and covariance based on the control input and measurement.
        
        Parameters:
            u (np.ndarray): Control input vector.
            y (np.ndarray): Measurement vector.
            hk (callable, optional): Updated measurement function. If None, use self.h.
            H_Jacobiank (callable, optional): Updated Jacobian of the measurement function. If None, use self.H_jacobian.
            Rk (np.ndarray, optional): Updated measurement noise covariance. If None, use self.R.
        """
        # Update the state measurement dynamics in case they are not constant
        if hk is not None:
            self.h = hk
        if H_Jacobiank is not None:
            self.H_jacobian = H_Jacobiank
        if Rk is not None:
            self.R = Rk

        # First, integrate the ODE to get x_hat_minus and P_minus
        x_hat_minus, P_minus = self.predict(u)

        self.k += 1  # Increment the time step

        # Update the filter
        tk = self.k * self.t_span
        H = self.H_jacobian(x_hat_minus, tk)
        M = self.M(x_hat_minus, tk)

        # Compute the innovation covariance
        G = M @ self.R @ M.T
        S = H @ P_minus @ H.T + G
        
        # Compute the Kalman gain
        self.Kalman = P_minus @ H.T @ np.linalg.inv(S)
        
        # Update the state estimate
        self.x_hat_plus = x_hat_minus + self.Kalman.dot(y - self.h(x_hat_minus, 0, self.t_span * self.k))
        
        # Update the estimation error covariance
        I = np.eye(self.xdim)  # Identity matrix of appropriate dimension
        if G.size == 1:
            left_term = self.Kalman @ self.Kalman.T * G
        else:
            left_term = self.Kalman @ G @ self.Kalman.T
        self.P_plus = (I - self.Kalman @ H) @ P_minus @ ((I - self.Kalman @ H).T) + left_term

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


class DiscreteExtendedKalmanFilter:
    def __init__(self, f, F_jac, L, h, H_jac, M, Q: np.ndarray, R: np.ndarray, x0: np.ndarray, P0: np.ndarray):
        """
        Initializes the Discrete Extended Kalman Filter with the provided functions and matrices.
        
        Parameters:
            f (callable): State transition function. Should take arguments (x, u, w, t) and return the state derivative.
            F_jacobian (callable): Function to compute the Jacobian of f. Should take arguments (x, u, w, t) and return the Jacobian matrix.
            L (callable): Function to compute df/dw at x_hat. Should take arguments (x, u, w, t) and return the matrix.
            h (callable): Measurement function. Should take arguments (x, v, t) and return the measurement.
            H_jacobian (callable): Function to compute the Jacobian of h. Should take arguments (x, t) and return the Jacobian matrix.
            M (callable): Function to compute dh/dw at x_hat. Should take arguments (x, t) and return the matrix.
            Q (np.ndarray): Process noise covariance matrix.
            R (np.ndarray): Measurement noise covariance matrix.
            x0 (np.ndarray): Initial state estimate vector.
            P0 (np.ndarray): Initial estimation error covariance matrix.
        """
        self.f = f
        self.F_Jac = F_jac
        self.L = L
        self.h = h
        self.H_jac = H_jac
        self.M = M
        self.Q = Q
        self.R = R
        self.x_hat_plus = x0
        self.P_plus = P0

        self.xdim = self.x_hat_plus.shape[0]
        self.k = 0

        self.Kalman = None
    
    def predict(self, u: np.ndarray):
        """
        Predict the next state of the dynamics based on the control vector u

        Parameters:
            u (np.ndarray): Control input vector.
        
        Returns:
            x_hat_minus (np.ndarray): The state estimation vector (pre-filtering).
            P_minus (np.ndarray): The state estimation error covariance (pre-filtering).
        """
        F = self.F_Jac(self.x_hat_plus, u)
        L = self.L(self.x_hat_plus, u)

        if self.xdim == 1:
            P_minus = self.P_plus * F @ F.T + L @ L.T * self.Q
        else:
            P_minus = F @ self.P_plus @ F.T + L @ self.Q @ L.T
        x_hat_minus = self.f(self.x_hat_plus, u, 0)

        return x_hat_minus, P_minus
    
    def update(self, u: np.ndarray, y: np.ndarray, fk = None, F_jack = None, Lk = None, hk = None, H_jack = None, Mk = None, Qk: np.ndarray = None, Rk: np.ndarray = None):
        """
        Updates the state estimate and covariance based on the control input and measurement.
        
        Parameters:
            u (np.ndarray): Control input vector.
            y (np.ndarray): Measurement vector.
            fk (callable, optional): Updated state transition function. If None, use self.f.
            F_jack (callable, optional): Updated Jacobian of the state transition function. If None, use self.F_Jac.
            Lk (callable, optional): Updated function to compute df/dw at x_hat. If None, use self.L.
            hk (callable, optional): Updated measurement function. If None, use self.h.
            H_jack (callable, optional): Updated Jacobian of the measurement function. If None, use self.H_jac.
            Mk (callable, optional): Updated function to compute dh/dw at x_hat. If None, use self.M.
            Qk (np.ndarray, optional): Updated process noise covariance matrix. If None, use self.Q.
            Rk (np.ndarray, optional): Updated measurement noise covariance matrix. If None, use self.R.
        """
        x_hat_minus, P_minus = self.predict(u)

        if fk is not None:
            self.f = fk
        if F_jack is not None:
            self.F_jac = F_jack
        if Lk is not None:
            self.L = Lk
        if hk is not None:
            self.h = hk
        if H_jack is not None:
            self.H_jac = H_jack
        if Mk is not None:
            self.M = Mk
        if Qk is not None:
            self.Q = Qk
        if Rk is not None:
            self.R = Rk
        
        H = self.H_jac(x_hat_minus)
        M = self.M(x_hat_minus)

        if self.R.shape[0] == 1:
            G = M @ M.T * self.R
        else:
            G = M @ self.R @ M.T

        if self.xdim == 1:
            S = P_minus * H @ H.T + G
            self.Kalman = P_minus * H.T @ np.linalg.inv(S)
            self.x_hat_plus = x_hat_minus + self.Kalman.dot(y - self.h(x_hat_minus, 0))
            self.P_plus = (np.eye(self.xdim) - self.Kalman @ H) * P_minus
        else:
            S = H @ P_minus @ H.T + G
            self.Kalman = P_minus @ H.T @ np.linalg.inv(S)
            self.x_hat_plus = x_hat_minus + self.Kalman.dot(y - self.h(x_hat_minus, 0))
            self.P_plus = (np.eye(self.xdim) - self.Kalman @ H) @ P_minus

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