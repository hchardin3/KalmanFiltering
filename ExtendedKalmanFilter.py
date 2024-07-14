import numpy as np
from scipy.integrate import odeint

class HybridExtendedKalmanFilter:
    def __init__(self, f, h, F_jacobian, H_jacobian, L, M, Q: np.ndarray, R: np.ndarray, P0: np.ndarray, x0: np.ndarray, t_span: float|int, precision: int=100):
        """
        Initializes the Extended Kalman Filter with the provided functions and matrices.
        
        Parameters:
            f (callable): State transition function. Should take arguments (x, u, w, t) and return the state derivative.
            h (callable): Measurement function. Should take arguments (x, v, t) and return the measurement.
            F_jacobian (callable): Function to compute the Jacobian of f. Should take arguments (x, u, w, t) and return the Jacobian matrix.
            H_jacobian (callable): Function to compute the Jacobian of h. Should take arguments (x, t) and return the Jacobian matrix.
            L (callable): Function to compute df/dw at x_hat. Should take arguments (x, u, w, t) and return the matrix.
            M (callable): Function to compute dh/dw at x_hat. Should take arguments (x, t) and return the matrix.
            Q (np.ndarray): Process noise covariance matrix.
            R (np.ndarray): Measurement noise covariance matrix.
            P0 (np.ndarray): Initial estimation error covariance matrix.
            x0 (np.ndarray): Initial state estimate vector.
            t_span (float|int): Time span for each integration step.
            precision (int): Number of points to use for the numerical integration.
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

        self.xdim = x0.shape[0]  # Dimension of the state vector
    
    def combined_derivatives(self, combined, t, u):
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
    
    def integrate_xhat_and_P(self, u: np.ndarray):
        """
        Integrates the state estimate and covariance from time (k-1)+ to time k-.
        
        Parameters:
            u (np.ndarray): Control input vector.
        """
        t_span = np.linspace(self.k * self.t_span, (self.k + 1) * self.t_span, self.precision)
        
        # Combine the initial state estimate and covariance
        combined_initial = np.hstack((self.x_hat_plus, self.P_plus.flatten()))
        
        # Integrate the combined state estimate and covariance
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
        x_hat_minus, P_minus = self.integrate_xhat_and_P(u)

        self.k += 1  # Increment the time step

        # Update the filter
        H = self.H_jacobian(x_hat_minus)
        M = self.M(x_hat_minus)

        # Compute the innovation covariance
        S = H.dot(P_minus).dot(H.T) + M.dot(self.R).dot(M.T)
        
        # Compute the Kalman gain
        self.Kalman = P_minus.dot(H.T).dot(np.linalg.inv(S))
        
        # Update the state estimate
        self.x_hat_plus = x_hat_minus + self.Kalman.dot(y - self.h(x_hat_minus, 0, self.t_span * self.k))
        
        # Update the estimation error covariance
        I = np.eye(self.xdim)  # Identity matrix of appropriate dimension
        self.P_plus = (I - self.Kalman.dot(H)).dot(P_minus).dot((I - self.Kalman.dot(H)).T) + self.Kalman.dot(M).dot(self.R).dot(M.T).dot(self.Kalman.T)

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
