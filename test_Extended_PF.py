import numpy as np
import math
from kalman_filters.ParticleFilter import ExtendedParticleFilter
from kalman_filters.PDF import MultivariateGaussianPDF, UnivariateGaussianPDF, MultivariateUniformPDF, UnivariateUniformPDF

def state_transition(x, w):
    return x * math.cos(x) + w

def state_transition_jacobian(x):
    return np.array([[np.cos(x) - x * np.sin(x)]])

def measurement_function(x, v):
    return x**2 + v

def measurement_jacobian(x):
    return np.array([[2 * x]])

# Noise settings
process_noise_std = 0.01
measurement_noise_std = 0.01

# Probability density functions for noise
dynamics_noise_pdf = UnivariateUniformPDF(lower_bound=-process_noise_std, upper_bound=process_noise_std)
measurement_noise_pdf = UnivariateGaussianPDF(mean=0, covariance=measurement_noise_std**2)
initial_state_pdf = UnivariateGaussianPDF(mean=1, covariance=0.1)

# Number of particles
N_particles = 10

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
true_state = np.ones(1)
num_steps = 100
true_states = [true_state]
positions = [true_state[0]]
measurements = []
estimates = [true_state]
for i in range(1, num_steps):
    print(i)
    
    true_state = state_transition(true_states[i-1], dynamics_noise_pdf.sample(1))
    true_states.append(true_state)
    positions.append(true_state[0])
    measurement = measurement_function(true_state, measurement_noise_pdf.sample(1))
    measurements.append(measurement)
    # print(type(measurement), measurement, measurement.shape)
    epf.update(measurement)
    estimates.append(epf.get_estimate())
    # print(estimates[i], true_state)

# Plot the results
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
states_plot = [true_state[0] for true_state in true_states]
plt.plot(states_plot, label='True State')
estimates_plot = [estimate[0] for estimate in estimates]
plt.plot(estimates_plot, label='Estimated State')
plt.xlabel('Time Step')
plt.ylabel('State')
plt.legend()
plt.grid()
plt.show()
