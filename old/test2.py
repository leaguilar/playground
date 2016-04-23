from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema



def simplify3(nk):
	nk = nk/np.sum(nk)
	#X_plot = np.linspace(0, len(nk), 1000)[:, np.newaxis]
	sdiv=100
	X_plot = np.linspace(0, len(nk), sdiv)[:, np.newaxis]
	custm = stats.rv_discrete(name='custm',a=0,b=7, values=(range(len(nk)), nk))
	yk= custm.rvs(size=1000)
	#yk.flatten()
	fig, ax = plt.subplots(1, 1)
	#ax.hist(yk, normed=True, histtype='stepfilled', alpha=0.2)
	# gaussian KDE
	X=yk.reshape(-1, 1)
	kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(X)
	log_dens = kde.score_samples(X_plot)
	mi, ma = argrelextrema(log_dens, np.less)[0], argrelextrema(log_dens, np.greater)[0]
	mi=np.rint(mi*float(len(nk))/float(sdiv))
	ma=np.rint(ma*float(len(nk))/float(sdiv))
	print "MIN ", mi
	print "MAX ", ma
	#print X_plot
	ax.plot(X_plot, np.exp(log_dens))
	ax.plot(X[:, 0], np.zeros(X.shape[0]) - 0.2, '+k')
	#ax.text(-3.5, 0.31, "Tophat Kernel Density")
	plt.show()

tk = np.array([0.1, 0.2, 0.3, 0.1, 0.1, 0.0, 0.5])
simplify3(tk)

