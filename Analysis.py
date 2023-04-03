import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
%matplotlib inline

x = np.array([5, 10]).reshape(-1, 1) # true position

mu = np.array([0, 0])
Ra = np.matrix([[9, 1],
               [1, 1]])
Rb = np.matrix([[1, 1],
                [1, 9]])

YA = []
YB = []
XML = []

for i in range(1000):
    ya = x + np.random.multivariate_normal(mu, Ra).reshape(-1, 1)
    yb = x + np.random.multivariate_normal(mu, Rb).reshape(-1, 1)
    xml = (Ra.I+Rb.I).I*(Ra.I*ya+Rb.I*yb)
    YA.append(ya.T)
    YB.append(yb.T)
    XML.append(xml.T)

YA = np.vstack(YA)
YB = np.vstack(YB)
XML = np.vstack(XML)

plt.figure(figsize=(10, 6))
plt.title('Finding Data for Two GPSs', fontsize=15)
plt.plot(YA[:,0], YA[:,1], 'b.', label='Observation 1')
plt.plot(YB[:,0], YB[:,1], 'r.', label='Observation 2')
plt.plot(XML[:,0], XML[:,1], 'k.', label='MLE')
plt.axis('equal')
plt.grid(alpha=0.3)
plt.legend(fontsize=15)
plt.show()
