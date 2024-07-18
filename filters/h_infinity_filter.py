import numpy as np

def is_positive_definite(matrix: np.ndarray):
    """
    Check if a given matrix is positive definite using the Cholesky decomposition.

    Parameters:
        matrix (np.ndarray): The matrix to check. Must be a square matrix.

    Returns:
        bool: True if the matrix is positive definite, False otherwise.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")
    
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


class DiscreteHInfinityFilter:
    def __init__(self, F: np.ndarray, H: np.ndarray, L: np.ndarray, Q: np.ndarray, R: np.ndarray, S: np.ndarray, x0: np.ndarray, P0: np.ndarray, theta: float = 1.0):
        """
        Initializes the H-infinity filter for systems with discrete dynamics.

        Parameters:
            F (np.ndarray): State transition matrix.
            H (np.ndarray): Measurement matrix.
            L (np.ndarray): Matrix for desired estimates.
            Q (np.ndarray): Process noise covariance matrix.
            R (np.ndarray): Measurement noise covariance matrix.
            S (np.ndarray): Cost function weight matrix.
            x0 (np.ndarray): Initial state estimate vector.
            P0 (np.ndarray): Initial estimation error covariance matrix.
            theta (float): H-infinity norm bound. A lower value leads to more robustness to model uncertainties. By default set to 1.
        """
        self.F = F
        self.H = H
        self.L = L
        self.Q = Q
        self.R = R
        self.S = S
        self.P = P0
        self.x_hat = x0
        self.theta = theta

    def update(self, y: np.ndarray, Fk: np.ndarray = None, Hk: np.ndarray = None, Lk: np.ndarray = None, Qk: np.ndarray = None, Rk: np.ndarray = None, Sk: np.ndarray = None):
        """
        Update the state estimate and covariance based on the measurement.

        Parameters:
            y (np.ndarray): Measurement vector.
            Other Parameters (nd.array, optional): Updates of state matrices. Optional
        """

        # Compute intermediate matrices
        S_tilt = self.L.T @ self.S @ self.L

        M = np.linalg.inv(np.eye(self.P.shape[0]) - self.theta * S_tilt @ self.P + self.H.T @ np.linalg.inv(self.R) @ self.H @ self.P)
        K_k = self.P @ M @ self.H.T @ np.linalg.inv(self.R)
        
        # Update state estimate
        self.x_hat = self.F @ self.x_hat + K_k @ (y - self.H @ self.x_hat)
        
        # Update covariance
        self.P = self.F @ self.P @ M @ self.F.T + self.Q

        # Update system matrices if provided
        if Fk is not None:
            self.F = Fk
        if Lk is not None:
            self.L = Lk
        if Hk is not None:
            self.H = Hk
        if Qk is not None:
            self.Q = Qk
        if Rk is not None:
            self.R = Rk
        if Sk is not None:
            self.S = Sk

        test = self.check_condition(S_tilt)

        if not test:
            print("DANGER: the filter might not be precise enough.")
    
    def check_condition(self, S_tilt: np.ndarray) -> bool:
        """
        Check the positive definiteness condition required for the filter's stability.

        Parameters:
            S_tilt (np.ndarray): Intermediate matrix derived from S and L.

        Returns:
            bool: True if the condition is satisfied, False otherwise.
        """
        to_test = np.linalg.inv(self.P) - self.theta * S_tilt + self.H.T @ np.linalg.inv(self.R) @  self.H
        return is_positive_definite(to_test)  
    
    def get_estimate(self) -> np.ndarray:
        """
        Returns the current state estimate.

        Returns:
            np.ndarray: The current state estimate.
        """
        return self.x_hat

    def get_error_covariance(self) -> np.ndarray:
        """
        Returns the current estimation error covariance.

        Returns:
            np.ndarray: The current estimation error covariance.
        """
        return self.P

