from stats.probability_density import *
import numpy as np

bounds = [(-10, 10) for _ in range(3)]

pdf1 = MultivariateUniformPDF(bounds)

print("Test MultUniform")
print(pdf1.mean)
print(pdf1.covariance_matrix)
print(pdf1.space_bounds)
print(pdf1.compute_mean_integral())
print(pdf1.evaluate(np.array([0, 0, 0])))
print(pdf1.sample(10).mean(axis=0))


print("Test UniUniform")
pdf1 = UnivariateUniformPDF(-1, 1)

print(pdf1.mean)
print(pdf1.covariance_matrix)
print(pdf1.space_bounds)
print(pdf1.compute_mean_integral())
print(pdf1.evaluate(0.6))
print(pdf1.sample(10).mean(axis=0))

print("Test UniGamma")
pdf2 = UnivariateGammaPDF(shape=2.0, scale=10.0)

print(pdf2.mean)
print(pdf2.covariance_matrix)
print(pdf2.space_bounds)
print(pdf2.compute_mean_integral())
print(pdf2.evaluate(20))
print(pdf2.sample(10).mean())

print("Test UniGaussian")
pdf3 = UnivariateGaussianPDF(0)

print(pdf3.mean)
print(pdf3.covariance_matrix)
print(pdf3.space_bounds)
print(pdf3.compute_mean_integral())
print(pdf3.evaluate(6))
print(pdf3.sample(10).mean())

print("Test MultiGaussian")
pdf3 = MultivariateGaussianPDF(mean=np.zeros(3))

print(pdf3.mean)
print(pdf3.covariance_matrix)
print(pdf3.space_bounds)
print(pdf3.compute_mean_integral())
print(pdf3.evaluate(np.array([20, 0, -0.5])))
print(pdf3.sample(10).mean())

print("Test UniExp")
pdf3 = UnivariateExponentialPDF()

print(pdf3.mean)
print(pdf3.covariance_matrix)
print(pdf3.space_bounds)
print(pdf3.compute_mean_integral())
print(pdf3.evaluate(1))
print(pdf3.sample(10).mean())

print("Test MultiExp")
pdf3 = MultivariateExponentialPDF(np.array([1, 1, 1]))

print(pdf3.mean)
print(pdf3.covariance_matrix)
print(pdf3.space_bounds)
print(pdf3.compute_mean_integral())
print(pdf3.evaluate(np.array([20, 0, -0.5])))
print(pdf3.sample(10).mean())