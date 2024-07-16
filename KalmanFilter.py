import numpy as np
from scipy.linalg import expm, inv, solve_continuous_are

class SimpleDiscreteKalmanFilter:
    def __init__(self, F: np.ndarray, G: np.ndarray, Q: np.ndarray, H: np.ndarray, R: np.ndarray, P0: np.ndarray, x0: np.ndarray):
        """
        Initializes the Kalman Filter with the provided matrices.
        
        Parameters:
            F (np.ndarray): State transition matrix.
            G (np.ndarray): Control input matrix.
            Q (np.ndarray): Process noise covariance.
            H (np.ndarray): Observation matrix.
            R (np.ndarray): Measurement noise covariance.
            P0 (np.ndarray): Initial estimation error covariance.
            x0 (np.ndarray): Initial state estimate.
        """
        self.F = F
        self.G = G
        self.Q = Q
        self.H = H
        self.R = R
        self.P_plus = P0
        self.x_hat_plus = x0
        self.xdim = x0.shape[0]
        self.ydim = H.shape[0]
        self.Kalman = None
    
    def predict(self, u):
        P_minus = self.F @ self.P_plus @ self.F.T + self.Q
        x_hat_minus = self.F.dot(self.x_hat_plus) + self.G.dot(u)
        return x_hat_minus, P_minus
    
    def update(self, u: np.ndarray, y: np.ndarray, Fk: np.ndarray = None, Gk: np.ndarray = None, Hk: np.ndarray = None, Qk: np.ndarray = None, Rk: np.ndarray = None):
        """
        Updates the state estimate and covariance based on the control input and measurement.
        
        Parameters:
            u (np.ndarray): Control input vector. 
            y (np.ndarray): Measurement vector.
            Fk, Gk, Hk, Qk, Rk (np.ndarray): Optional updated system matrices.
        """
        # Prediction step
        x_hat_minus, P_minus = self.predict(u)

        # Update system matrices if provided
        if Fk is not None:
            self.F = Fk
        if Gk is not None:
            self.G = Gk
        if Hk is not None:
            self.H = Hk
        if Qk is not None:
            self.Q = Qk
        if Rk is not None:
            self.R = Rk
        
        # Update step
        S = self.H @ P_minus @ self.H.T + self.R
        self.Kalman = P_minus @ self.H.T @ np.linalg.inv(S)
        self.x_hat_plus = x_hat_minus + self.Kalman.dot(y - self.H.dot(x_hat_minus))
        self.P_plus = (np.eye(self.xdim) - self.Kalman @ self.H) @ P_minus
    
    def get_estimate(self) -> np.ndarray:
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

class SimpleContinuousKalmanFilter(SimpleDiscreteKalmanFilter):
    def __init__(self, dt: float, A: np.ndarray, B: np.ndarray, Q: np.ndarray, H: np.ndarray, R: np.ndarray, P0: np.ndarray, x0: np.ndarray):
        """
        Initializes the continuous-to-discrete Kalman Filter with the provided matrices and sampling time.
        
        Parameters:
            dt (float): Sampling time.
            A (np.ndarray): Continuous-time state transition matrix.
            B (np.ndarray): Continuous-time control input matrix.
            Q (np.ndarray): Process noise covariance.
            H (np.ndarray): Observation matrix.
            R (np.ndarray): Measurement noise covariance.
            P0 (np.ndarray): Initial estimation error covariance.
            x0 (np.ndarray): Initial state estimate.
        """
        
        F = expm(A * dt)

        # Check if A is invertible
        if np.linalg.matrix_rank(A) == A.shape[0]:
            # Compute G using the matrix inversion method
            G = inv(A).dot(F - np.eye(A.shape[0])).dot(B)
        else:
            # Compute G using the block matrix method
            # Construct the augmented matrix
            n = A.shape[0]
            Z = np.zeros_like(A)
            augmented_matrix = np.block([[A, B],
                                        [Z, Z]])
            # Compute the matrix exponential of the augmented matrix
            exp_augmented = expm(augmented_matrix * dt)
            # Extract G from the result
            G = exp_augmented[:n, n:]

        super().__init__(F, G, Q, H, R, P0, x0)

class ContinuousTimeKalmanFilter:
    def __init__(self, A, B, C, Qc, Rc, P0, x0):
        """
        Initializes the Continuous Kalman Filter with the provided matrices.
        
        Parameters:
            A (np.ndarray): State transition matrix.
            B (np.ndarray): Control input matrix.
            C (np.ndarray): Observation matrix.
            Qc (np.ndarray): Process noise covariance.
            Rc (np.ndarray): Measurement noise covariance.
            P0 (np.ndarray): Initial estimation error covariance.
            x0 (np.ndarray): Initial state estimate.
        """
        self.A = A
        self.B = B
        self.C = C
        self.Qc = Qc
        self.Rc = Rc
        self.P_plus = P0
        self.x_hat_plus = x0

    def _state_derivative(self, x_hat, u, Kalman, y):
        """
        Computes the derivative of the state estimate.
        """
        return self.A @ x_hat + self.B @ u + Kalman @ (y - self.C @ x_hat)

    def _covariance_derivative(self, P, Kalman):
        """
        Computes the derivative of the error covariance matrix.
        """
        return -Kalman @ self.C @ P + self.A @ P + P @ self.A.T + self.Qc

    def _runge_kutta_4(self, f, y0, dt, *args):
        """
        Fourth-order Runge-Kutta integration method.
        """
        k1 = f(y0, *args)
        k2 = f(y0 + 0.5 * dt * k1, *args)
        k3 = f(y0 + 0.5 * dt * k2, *args)
        k4 = f(y0 + dt * k3, *args)
        return y0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def update(self, u, y, dt):
        """
        Updates the state estimate and covariance matrix using the Kalman filter equations.
        
        Parameters:
            u (np.ndarray): Control input vector.
            y (np.ndarray): Measurement vector.
            dt (float): Time step for numerical integration.
        """
        # Compute the Kalman gain
        S = self.C @ self.P @ self.C.T + self.Rc
        Kalman = self.P @ self.C.T @ np.linalg.inv(S)
        
        # Update the state estimate using RK4 integration
        self.x_hat_plus = self._runge_kutta_4(self._state_derivative, self.x_hat_plus, dt, u, Kalman, y)
        
        # Update the error covariance matrix using RK4 integration
        self.P_plus = self._runge_kutta_4(self._covariance_derivative, self.P_plus, dt, Kalman)

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