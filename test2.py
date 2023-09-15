import numpy as np
import matplotlib.pyplot as plt

# Parameters for the Gaussian distributions
mean = 0
variance1 = 1
variance2 = 1.36

# Generate data points for the x-axis
x = np.linspace(-3, 3, 1000)

# Calculate the probability density function (PDF) for both distributions
pdf1 = (1 / (np.sqrt(2 * np.pi * variance1))) * np.exp(-((x - mean)**2) / (2 * variance1))
pdf2 = (1 / (np.sqrt(2 * np.pi * variance2))) * np.exp(-((x - mean)**2) / (2 * variance2))

# Create a Matplotlib figure and plot the two Gaussian distributions
plt.figure(figsize=(8, 5))
plt.plot(x, pdf1, label=f'Variance = {variance1}')
plt.plot(x, pdf2, label=f'Variance = {variance2}')
plt.title('Two Gaussian Distributions with Mean 0')
plt.xlabel('X')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.savefig('gaussian_distributions.png')