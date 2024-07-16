import numpy as np
from kalman_filters.ParticleFilter import ExtendedParticleFilter, GaussianPDF

def state_transition(x, w):
    return x + w

def state_transition_jacobian(x):
    return np.array([[1]])

def measurement_function(x, v):
    return x + v

def measurement_jacobian(x):
    return np.array([[1]])

# Noise settings
process_noise_std = 0.1
measurement_noise_std = 0.1

# Probability density functions for noise
dynamics_noise_pdf = GaussianPDF(mean=np.zeros(1), covariance_matrix=np.array([[process_noise_std ** 2]]))
measurement_noise_pdf = GaussianPDF(mean=np.zeros(1), covariance_matrix=np.array([[measurement_noise_std ** 2]]))

# Initial state PDF
initial_state_pdf = GaussianPDF(mean=np.array([0]), covariance_matrix=np.array([[1]]))

# Number of particles
N_particles = 1000

# Create the filter
epf = ExtendedParticleFilter(
    f=state_transition,
    F_jac=state_transition_jacobian,
    h=measurement_function,
    H_jac=measurement_jacobian,
    N_particles=N_particles,
    dynamics_noise_pdf=dynamics_noise_pdf,
    measurement_noise_pdf=measurement_noise_pdf,
    x0_pdf=initial_state_pdf
)

# Simulate some data
true_state = 0
num_steps = 20
true_states = []
measurements = []
for _ in range(num_steps):
    true_state += np.random.normal(0, process_noise_std)
    true_states.append(true_state)
    measurement = true_state + np.random.normal(0, measurement_noise_std)
    measurements.append(measurement)

# Run the particle filter
estimates = []
for measurement in measurements:
    epf.update(measurement)
    estimates.append(epf.get_estimate())

# Plot the results
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(true_states, label='True State')
plt.plot(estimates, label='Estimated State')
plt.xlabel('Time Step')
plt.ylabel('State')
plt.legend()
plt.show()
