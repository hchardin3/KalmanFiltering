from filters.particle_filters import ParticleFilter
from stats.univariate_pdf import UnivariateGaussianPDF, UnivariateUniformPDF
from stats.multivariate_pdf import MultivariateGaussianPDF, MultivariateUniformPDF
import numpy as np

# Define the state transition and measurement functions
def f(x, w):
    dt = 1  # time step
    F = np.array([[1, dt],
                  [0, 1]])
    return F @ x + w

def h(x, v):
    return x[1] * x[0] + v


# Parameters
N_particles = 10
dt = 1  # Time step
initial_position = 0
initial_velocity = 1  # Moving with constant velocity
initial_state = np.array([initial_position, initial_velocity])
initial_covariance = 0.001 * np.eye(2)

# Dynamics and measurement noise
process_noise_std = 1e-1
measurement_noise_std = 1e-3

# dynamics_noise_pdf = MultivariateGaussianPDF(mean=np.zeros(2), covariance_matrix=np.eye(2) * process_noise_std**2)
dynamics_noise_pdf = MultivariateUniformPDF(bounds=[(-process_noise_std, process_noise_std) for _ in range(2)])
measurement_noise_pdf = UnivariateUniformPDF(-measurement_noise_std, measurement_noise_std)

# Initialize Particle Filter
pf = ParticleFilter(f, h, N_particles, dynamics_noise_pdf, measurement_noise_pdf, x0=initial_state, P0=initial_covariance)

# Simulate over time
steps = 100
real_state = [initial_state]
real_position = [0]
# positions = [initial_position]
positions = [initial_position]
measurements = [np.array([initial_position, initial_velocity**2])]
for k in range(1, steps):
    w = dynamics_noise_pdf.sample(1)
    real_state.append(f(real_state[k-1], w))
    real_position.append(real_state[k][0])
    print(k)
    # No input (control) assumed
    y = h(real_state[k], measurement_noise_pdf.sample(1))
    measurements.append(y)
    pf.update(y)
    
    # Estimate and store position
    estimated_position = pf.get_estimate()[0]
    positions.append(estimated_position)
    

# Plotting results
import matplotlib.pyplot as plt

time = np.arange(steps)
plt.figure(figsize=(10, 5))
plt.plot(time, real_position, label='Real Position')
plt.plot(time, positions, label='Estimated Position')
measurements_plot = [measurements[k][0] for k in range(steps)]
# plt.plot(time, measurements_plot, 'r*', label='Measurements')
plt.title('Particle Filter State Estimation')
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.legend()
plt.grid(True)
plt.show()
