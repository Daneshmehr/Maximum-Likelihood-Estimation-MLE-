import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
%matplotlib inlin

# mean is unknown in this example
# variance is known in this example

m = 10
mu = 0
sigma = 5

x = np.random.normal(mu,sigma,[m,1])

mus = np.arange(-10, 10.5, 0.5)
LOGL = []

for i in range(np.size(mus)):
    y = norm.pdf(x, mus[i], sigma)
    logL = np.sum(np.log(y))
    LOGL.append(logL)

muhat = np.mean(x)
print(muhat)
    
plt.figure(figsize=(10, 6))
plt.plot(mus, LOGL, '.')
plt.title('$log (\prod \mathcal{N}(x \mid \mu , \sigma^2))$', fontsize=20)
plt.xlabel(r'$\hat \mu$', fontsize=15)
plt.grid(alpha=0.3)
plt.show()
