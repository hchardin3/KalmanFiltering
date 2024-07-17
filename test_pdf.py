from kalman_filters.ParticleFilter import ProbabilityDensityFunction
import numpy as np

def f(x:float):
    return 0.5 if abs(x) <= 1 else 0.

pdf1 = ProbabilityDensityFunction(f, 1, pdf_bounds=2)

print(pdf1.mean)
print(pdf1.covariance_matrix)