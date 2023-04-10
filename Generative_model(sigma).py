import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
%matplotlib inline

# mean is known in this example
# variance is unknown in this example

m = 100
mu = 0
sigma = 3

x = np.random.normal(mu,sigma,[m,1])  # samples

sigmas = np.arange(1, 10, 0.1)
LOGL = []

for i in range(sigmas.shape[0]):
    y = norm.pdf(x, mu, sigmas[i])     # likelihood
    logL = np.sum(np.log(y))
    LOGL.append(logL)

sigmahat = np.sqrt(np.var(x))
print(sigmahat)

plt.figure(figsize=(10,6))
plt.title(r'$\log (\prod \mathcal{N} (x|\mu,\sigma^2))$',fontsize=20)
plt.plot(sigmas, LOGL, '.')
plt.xlabel(r'$\hat \sigma$', fontsize=15)
plt.axis([0, np.max(sigmas), np.min(LOGL), -200])
plt.grid(alpha=0.3)
plt.show()
