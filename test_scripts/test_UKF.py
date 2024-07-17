from filters.unscented_kalman_filter import UnscentedKalmanFilter
import numpy as np
import matplotlib.pyplot as plt

# Define the dynamics model of the system
def dynamics(state, control_input, t):
    position, velocity = state
    acceleration = control_input
    dt = t  # assuming t represents the time step in seconds
    new_position = position + velocity * dt + 0.5 * acceleration * dt**2
    new_velocity = velocity + acceleration * dt
    return np.array([new_position, new_velocity])

# Define the measurement model of the system
def measurement_model(state, t):
    position, _ = state
    return np.array([position])

# Test parameters
initial_state = np.array([0, 0])  # starting at the origin with zero velocity
initial_covariance = np.eye(2) * 0.1  # small initial uncertainty
process_noise_cov = np.array([[0.1, 0.01], [0.01, 0.1]])  # position and velocity process noise
measurement_noise_cov = np.array([[0.1]])  # measurement noise
control_input = 1.0  # constant acceleration
dt = 1  # time step in seconds
num_steps = 300  # number of time steps for the simulation (300 seconds)

# Initialize the Unscented Kalman Filter
ukf = UnscentedKalmanFilter(f=dynamics, h=measurement_model, Q=process_noise_cov, R=measurement_noise_cov, P0=initial_covariance, x0=initial_state)

# Simulate the movement and filtering
true_states = [initial_state]
measurements = []
estimates = [initial_state]

for k in range(1, num_steps + 1):
    true_state = dynamics(true_states[-1], control_input, dt)
    true_states.append(true_state)
    
    measurement_noise = np.random.normal(0, np.sqrt(measurement_noise_cov[0,0]))
    measurement = measurement_model(true_state, dt) + measurement_noise
    measurements.append(measurement)
    
    ukf.update(measurement, control_input, dt)
    estimates.append(ukf.get_estimate())

true_states = np.array(true_states)
measurements = np.array(measurements)
estimates = np.array(estimates)

# Calculate RMSE
rmse = np.sqrt(np.mean((estimates - true_states)**2, axis=0))
print("\nRMSE for Position:", rmse[0])
print("RMSE for Velocity:", rmse[1])

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(range(num_steps + 1), true_states[:, 0], label='True Position', linewidth=2)
plt.scatter(range(1, num_steps + 1), [m[0] for m in measurements], color='red', label='Measured Position', alpha=0.6, s=8)
plt.title('True Position vs Measured Position')
plt.xlabel('Time (seconds)')
plt.ylabel('Position')
plt.legend()
plt.grid(True)
plt.show()
