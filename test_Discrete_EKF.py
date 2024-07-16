import numpy as np
from kalman_filters.ExtendedKalmanFilter import DiscreteExtendedKalmanFilter

# State transition function
def f(x, u, w=0, dt=1):
    return np.array([
        x[0] + x[1] * dt + 0.5 * u[0] * dt**2,
        x[1] + u[0] * dt
    ])

# Jacobian of the state transition function
def F_jac(x, u, w=0, dt=1):
    return np.array([
        [1, dt],
        [0, 1]
    ])

# df/dw
def L(x, u, w=0, dt=1):
    return np.eye(2)  # Assuming noise affects both position and velocity directly

# Measurement function
def h(x, v=0, t=0):
    return np.array([x[0] + x[1] + v])

# Jacobian of the measurement function
def H_jac(x, t=0):
    return np.array([[1, 1]])

# dh/dv
def M(x, t=0):
    return np.array([1])

# Initial conditions
x0 = np.array([0, 0])
P0 = np.eye(2) * 0.1
Q = np.eye(2) * 0.01
R = np.array([[0.1]])

# Create the filter
dt = 1
ekf = DiscreteExtendedKalmanFilter(f, F_jac, L, h, H_jac, M, Q, R, x0, P0)

# Simulation parameters
steps = 10
u = np.array([0.5])  # Constant acceleration
measurements = []
estimates = []
true_positions = []

# Simulate the system
for k in range(steps):
    # Simulate measurement
    x_true = f(ekf.get_estimate(), u, dt=dt)
    true_positions.append(x_true)
    measurement = h(x_true, np.random.normal(0, np.sqrt(R[0,0])))
    measurements.append(measurement)
    
    # Update the filter
    ekf.update(u, measurement)
    estimates.append(ekf.get_estimate())

# Plotting the results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot([true[0] for true in true_positions], label='Estimated Position')
plt.plot([est[0] for est in estimates], label='Estimated Position')
plt.scatter(range(len(measurements)), [m[0] for m in measurements], color='red', label='Measurements')
plt.legend()
plt.title('Discrete Extended Kalman Filter Performance')
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.grid(True)
plt.show()
