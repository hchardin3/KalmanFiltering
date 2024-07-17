from kalman_filters.ExtendedKalmanFilter import HybridExtendedKalmanFilter, DiscreteExtendedKalmanFilter

import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt

# Dynamics model
def f_vehicle(x, u, w, t):
    # Ensure u and w are arrays of the same shape
    u = np.asarray(u)  # Make sure u is an array, in case it isn't
    w = np.asarray(w)  # Same for w
    print("x: ", x, " | u: ", u, " | w: ", w)
    print(type(w))
    a =  np.array([x[1], u[0]]) 
    b = np.array([0., w])  # Use u[0] + w[0] to access the first element if they are arrays
    return a + b


# Measurement model
def h_vehicle(x, v, t):
    return np.array([x[0] + v])

# Jacobians
def F_jac_vehicle(x, u, w, t):
    return np.array([[0, 1], [0, 0]])

def H_jac_vehicle(x, t):
    return np.array([[1, 0]])

def L_vehicle(x, u, w, t):
    return np.array([0, 1])

def M_vehicle(x, t):
    return np.array([1])

# Initial conditions
x0_vehicle = np.array([0, 0])  # Initial position and velocity
P0_vehicle = np.eye(2)  # Initial covariance
Q_vehicle = np.array([[0, 0], [0, 1]])  # Process noise covariance (mostly in acceleration)
R_vehicle = np.array([[1]])  # Measurement noise covariance

# Create filter instance
vehicle_filter = HybridExtendedKalmanFilter(f=f_vehicle, h=h_vehicle, F_jacobian=F_jac_vehicle, H_jacobian=H_jac_vehicle,
                                            L=L_vehicle, M=M_vehicle, Q=Q_vehicle, R=R_vehicle, x0=x0_vehicle,   P0=P0_vehicle, t_span=1)

# Simulation parameters
t_final = 50
dt = 1
steps = int(t_final / dt)
times = np.linspace(0, t_final, steps)
measurements = []
estimates = []
true_positions = []

# Generate noise for simulation (if not dynamically in 'f_vehicle')
process_noise = np.random.normal(0, np.sqrt(Q_vehicle[1, 1]), size=(steps,))

# Modify the update loop to handle noise correctly
for k in range(steps):
    u = 0.5  # Constant acceleration
    noise = process_noise[k]  # Process noise as a scalar
    u_array = np.array([u])  # Convert scalar to array
    noise_array = np.array([noise])  # Convert scalar to array
    
    # Simulate measurement with the control and noise
    true_position = vehicle_filter.x_hat_plus[0] + vehicle_filter.x_hat_plus[1] * dt + 0.5 * (u + noise) * dt**2
    true_positions.append(true_position)
    measured_position = true_position + np.random.normal(0, np.sqrt(R_vehicle[0, 0]))
    measurements.append(measured_position)
    
    # Call update with the combined control input and noise
    vehicle_filter.update(u_array + noise_array, np.array([measured_position]))
    
    # Append the current estimated state to the estimates list
    estimates.append(vehicle_filter.get_estimate())

# Now the plotting section should work as intended
plt.figure(figsize=(10, 5))
plt.plot(times, true_positions, label='True Position')
plt.scatter(times, measurements, color='red', label='Measured Position', alpha=0.6)
plt.plot(times, [est[0] for est in estimates], label='Estimated Position', linestyle='--')
plt.xlabel('Time (seconds)')
plt.ylabel('Position')
plt.title('Hybrid Extended Kalman Filter: Vehicle Position Estimation')
plt.legend()
plt.grid(True)
plt.show()
