import numpy as np
from scipy.stats import norm

data = np.array([[11.8, 0.5, 0.5], [6.3, 0.2, 0.8], [19.1, 0.8, 0.2], [10.9, 0.4, 0.6], [19.2, 0.8, 0.2],
                 [4.3, 0.2, 0.8], [14.4, 0.6, 0.4], [4.6, 0.2, 0.8], [13, 0.5, 0.5], [12.6, 0.5, 0.5]])
mu1 = 0
mu2 = 0
o1 = 0
o2 = 0
for i in data:
    mu1 += i[0] * i[1]
    o1 += i[1]
    mu2 += i[0] * i[2]
    o2 += i[2]
mu1 = mu1 / o1
mu2 = mu2 / o2
print(mu1, mu2)

mu1 = 13
mu2 = 9.7
o1 = 4.1
o2 = 4.2
P1 = 0.5
P2 = 0.5
x = 12.8

Pxc1 = norm.pdf(x, mu1, o1)
Pxc2 = norm.pdf(x, mu2, o2)

Pcx1 = (Pxc1 * P1) / (Pxc1 * P1 + Pxc2 * P2)
print(Pcx1)
Pcx2 = (Pxc2 * P2) / (Pxc1 * P1 + Pxc2 * P2)
print(Pcx2)
