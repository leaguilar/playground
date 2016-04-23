#!/usr/bin/env python

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.neighbors.kde import KernelDensity
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

from scipy import stats
from scipy.stats import norm



def simplify_data1(x):

	X = np.array(zip(x,np.zeros(len(x))), dtype=np.float)
	bandwidth = estimate_bandwidth(X, quantile=0.1)
	ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
	ms.fit(X)
	labels = ms.labels_
	cluster_centers = ms.cluster_centers_

	labels_unique = np.unique(labels)
	n_clusters_ = len(labels_unique)
	print X
	start=0
	for k in range(n_clusters_):
	    my_members = labels == k
	    print "cluster {0}: {1}".format(k, X[my_members, 0]),np.average(X[my_members, 0])
	    for i in xrange(start,start+len(X[my_members, 0])):
		print i
		X[i][0]=np.average(X[my_members, 0])
	    start+=len(X[my_members, 0])
	return X[:,0]

def simplify_data2(x,y,size):
	avg=[]
	result=[]
	kde = KernelDensity(kernel='tophat', bandwidth=0.5).fit(x)
	s = np.linspace(0,size,len(x))
	e = kde.score_samples(s.reshape(-1,1))
	mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
	start=0	
	for i in mi:
		val=np.average(x[start:i])
		for j in xrange(start,i):
			result.append(val)
		start=i	
	val=np.average(x[start:])
	for j in xrange(start,len(x)):
			result.append(val)
	#plt.plot(s, e*0.01+e[mi[0]])
	print mi
	print ma
	plt.plot(s,x.reshape(1,-1)[0])
	plt.plot(s,result)
	#print x, len(x)
	plt.show()


def simplify3(nk):
	result=[]
	nk=np.array(nk)
	nk = nk/float(np.sum(nk))
	print nk
	
	#X_plot = np.linspace(0, len(nk), 1000)[:, np.newaxis]
	sdiv=1000
	X_plot = np.linspace(0, len(nk), sdiv)[:, np.newaxis]
	custm = stats.rv_discrete(name='custm',a=0,b=7, values=(range(len(nk)), nk))
	yk= custm.rvs(size=100000)
	#yk.flatten()
	fig, ax = plt.subplots(1, 1)
	ax.hist(yk, normed=True, histtype='stepfilled', alpha=0.2)
	# gaussian KDE
	X=yk.reshape(-1, 1)
	kde = KernelDensity(kernel='gaussian', bandwidth=0.3).fit(X)
	log_dens = kde.score_samples(X_plot)
	mi, ma = argrelextrema(log_dens, np.less)[0], argrelextrema(log_dens, np.greater)[0]
	mi=np.rint(mi*float(len(nk))/float(sdiv))
	ma=np.rint(ma*float(len(nk))/float(sdiv))
	start=0	
	print mi
	for i in mi:
		i=int(i)
		if start!=i:
			val=np.average(nk[start:i])
			for j in xrange(start,i):
				result.append(val)
		start=i	
	val=np.average(nk[start:])
	for j in xrange(start,len(nk)):
			result.append(val)
	#plt.plot(s, e*0.01+e[mi[0]])
	print mi
	print ma
	print nk
	#plt.plot(range(len(nk)),nk)
	plt.plot(range(len(result)),result)
	#print x, len(x)
	plt.show()


x = [1,1,1,1,1,5,5,5,5,1,1,5,5]

simplify3(x)



