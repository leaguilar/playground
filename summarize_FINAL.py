#!/usr/bin/env python

from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import os

from itertools import permutations
from random import sample


import numpy as np

from scipy import stats
from scipy.stats import norm
from sklearn.neighbors.kde import KernelDensity
from scipy.signal import argrelextrema

def simplify3(nk):
	result=[]
	nk=np.array(nk)
	if not float(np.sum(nk)):
		return nk	
	xk = nk/float(np.sum(nk))
	#print nk
	
	#X_plot = np.linspace(0, len(nk), 1000)[:, np.newaxis]
	sdiv=1000
	X_plot = np.linspace(0, len(xk), sdiv)[:, np.newaxis]
	custm = stats.rv_discrete(name='custm',a=0,b=7, values=(range(len(xk)), xk))
	yk= custm.rvs(size=100000)
	#yk.flatten()
	#fig, ax = plt.subplots(1, 1)
	#ax.hist(yk, normed=True, histtype='stepfilled', alpha=0.2)
	# gaussian KDE
	X=yk.reshape(-1, 1)
	kde = KernelDensity(kernel='gaussian', bandwidth=0.6).fit(X)
	log_dens = kde.score_samples(X_plot)
	mi, ma = argrelextrema(log_dens, np.less)[0], argrelextrema(log_dens, np.greater)[0]
	mi=np.rint(mi*float(len(xk))/float(sdiv))
	ma=np.rint(ma*float(len(xk))/float(sdiv))
	start=0	
	#print mi
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
	return np.array(result)
	#plt.plot(s, e*0.01+e[mi[0]])
	#print mi
	#print ma
	#print nk
	#plt.plot(range(len(nk)),nk)
	#plt.plot(range(len(result)),result)
	#print x, len(x)
	#plt.show()
	#quit()
	

def process_file(data_path,filename):
		fin = open(data_path+filename, 'r');
		ex_data = np.genfromtxt(fin, delimiter=',', skip_header=0,skip_footer=0, names=['day','time', 'value']);
		color_idx=int(ex_data['day'][0])		
		fin.close()
		#plot_best(ex_data,"-","gray","",sub1)
		print "OPENING ",filename,int(ex_data['day'][0])
		startIdx=0;
		endIdx=0;
		color_idx=0
		for i in xrange(1,ex_data['day'].size):
			if ex_data['day'][i-1]!=ex_data['day'][i]:
				endIdx=i-1
				#Winter
				#if int(ex_data['day'][startIdx]) <= 240 and int(ex_data['day'][startIdx]) >= 240:
				#	#print startIdx,endIdx,ex_data['time'][startIdx],ex_data['time'][endIdx]
				#	plot_simple(ex_data,"blue",sub1,startIdx,endIdx)
				#	new_val=simplify3(ex_data['value'][startIdx:endIdx])
				#	for j in xrange(startIdx,endIdx):
				#		ex_data['value'][j]=new_val[j-startIdx]
				#	print new_val
				#	print ex_data['value'][startIdx:endIdx]
				#	#print startIdx,endIdx,ex_data['time'][startIdx],ex_data['time'][endIdx]
				#	plot_simple(ex_data,"green",sub1,startIdx,endIdx)
				#Winter
				#if int(ex_data['day'][startIdx]) <= 240 and int(ex_data['day'][startIdx]) >= 240:
				#print startIdx,endIdx,ex_data['time'][startIdx],ex_data['time'][endIdx]
				#plot_simple(ex_data,"blue",sub1,startIdx,endIdx)
				new_val=simplify3(ex_data['value'][startIdx:endIdx])
				for j in xrange(startIdx,endIdx):
					ex_data['value'][j]=new_val[j-startIdx]
				#print new_val
				#print ex_data['value'][startIdx:endIdx]
				#print startIdx,endIdx,ex_data['time'][startIdx],ex_data['time'][endIdx]
				#plot_simple(ex_data,"green",sub1,startIdx,endIdx)
				startIdx=endIdx+1
				
				#plot_simple(ex_data,color_idx,color_idx,sub1,startIdx,endIdx)
		np.savetxt(data_path_out+filename, ex_data, delimiter=",",fmt='%i,%i,%.3f')
		#if int(ex_data['day'][0]) < 240 and int(ex_data['day'][0]) > 213:
		#	plot_simple(ex_data,color_idx,color_idx,sub1)
		#print "FINISHED ",filename

		#exit()

# set directory and input *.csv file names
path_out="./"
out_name="learning_result.csv"

data_path = '../raw/'
data_path_out = '../output/'
target_name="user_"

startIdx=0;
endIdx=0;

count=0

lfiles=[]
for filename in os.listdir(data_path):
	#count+=1
	#if count >1:
	#	break
	if target_name in filename:
		lfiles.append(filename)
	
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

my_size=int(float(len(lfiles))/float(size))

if rank == size-1:
	my_size+=len(lfiles)-my_size*size

istart=rank*my_size
iend=(1+rank)*my_size

print "my rank", rank, "size",size
print istart, iend

for current in lfiles[istart:iend]:
	process_file(data_path,current)

exit()
