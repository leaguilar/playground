from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

N = 20
#X = np.concatenate((np.random.normal(0, 1, 0.3 * N), np.random.normal(5, 1, 0.7 * N)))[:, np.newaxis]

xk = np.arange(7)
nk = np.array([0.1, 0.2, 0.3, 0.1, 0.1, 0.0, 0.2])
nk = nk/np.sum(nk)
pk = (0.1, 0.2, 0.3, 0.1, 0.1, 0.0, 0.2)

print totuple(nk)

custm = stats.rv_discrete(name='custm', values=(xk, nk))

np.random.seed(282629734)
x = custm.rvs(size=1000)


fig, ax = plt.subplots(1, 1)
#ax.plot(xk, custm.pmf(xk), 'ro', ms=12, mec='r')
#ax.vlines(xk, 0, custm.pmf(xk), colors='r', lw=4)
ax.hist(x, normed=False, histtype='stepfilled', alpha=0.2)
plt.show()
exit()
