from kalman_filters.ParticleFilter import ParticleFilter, GaussianPDF
import numpy as np

# Define the state transition and measurement functions
def f(x, w):
    dt = 1  # time step
    F = np.array([[1, dt],
                  [0, 1]])
    return F @ x + w

def h(x, v):
    H = np.array([1, 0])
    return H @ x + v

# Parameters
N_particles = 1000
dt = 1  # Time step
initial_position = 0
initial_velocity = 1  # Moving with constant velocity
initial_state = np.array([initial_position, initial_velocity])
initial_covariance = np.eye(2) * 0.1

# Dynamics and measurement noise
process_noise_std = 0.1
measurement_noise_std = 0.5

dynamics_noise_pdf = GaussianPDF(mean=np.zeros(2), covariance_matrix=np.eye(2) * process_noise_std**2)
measurement_noise_pdf = GaussianPDF(mean=np.zeros(1), covariance_matrix=np.array([[measurement_noise_std**2]]))

# Initialize Particle Filter
pf = ParticleFilter(f, h, N_particles, dynamics_noise_pdf, measurement_noise_pdf, x0=initial_state, P0=initial_covariance)

# Simulate over time
steps = 20
real_state = [np.array([0, 1])]
real_position = [0]
positions = [initial_position]
for k in range(1, steps):
    w = dynamics_noise_pdf.sample(1)
    real_state.append(f(real_state[k-1], w))
    real_position.append(real_state[k][0])
    print(k)
    # No input (control) assumed
    y = h(initial_state, np.random.normal(0, measurement_noise_std))
    pf.update(y)
    
    # Estimate and store position
    estimated_position = pf.get_estimate()[0]
    positions.append(estimated_position)
    
    # Update the true state (for simulation purposes)
    initial_state = f(initial_state, np.random.normal(0, process_noise_std, 2))

# Plotting results
import matplotlib.pyplot as plt

time = np.arange(steps)
plt.figure(figsize=(10, 5))
plt.plot(time, real_position, label='Real Position')
plt.plot(time, positions, label='Estimated Position')
plt.title('Particle Filter State Estimation')
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.legend()
plt.grid(True)
plt.show()
