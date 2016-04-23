#!/usr/bin/env python

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import textwrap
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import os

from itertools import permutations
from random import sample


import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

from scipy import stats
from scipy.stats import norm
from sklearn.neighbors.kde import KernelDensity
from scipy.signal import argrelextrema

def simplify_data1(x):
	X = np.array(zip(x,np.zeros(len(x))), dtype=np.float)
	bandwidth = estimate_bandwidth(X, quantile=0.2)
	ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
	ms.fit(X)
	labels = ms.labels_
	cluster_centers = ms.cluster_centers_
	labels_unique = np.unique(labels)
	n_clusters_ = len(labels_unique)
	#print n_clusters_
	#exit()
	start=0
	value=0
	print x
	for k in range(n_clusters_):
	    my_members = labels == k
	    print "cluster {0}: {1}".format(k, X[my_members, 0]),np.average(X[my_members, 0])
	    value=np.average(X[my_members, 0])
	    val2=0
	    for i in xrange(start,start+len(X[my_members, 0])):
		val2+=X[i][0]
		print val2,X[i][0],i
		X[i][0]=value
	    print "FINAL",val2/len(X[my_members, 0])
	    start+=len(X[my_members, 0])
	return X[:,0]

def simplify3(nk):
	result=[]
	nk=np.array(nk)
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
	
lines = ["-","--","-.",":"]
linecycler = cycle(lines)
label_texts = list()
#lines = ["-"]

#Get the color-wheel
Nlines = 100
color_lvl = 16
rgb = np.array(list(permutations(range(0,256,color_lvl),3)))/255.0
colors = sample(rgb,Nlines)

print len(rgb)
print len(colors)


def addtoDict(key,value,aDict):
    if not key in aDict:
        aDict[key] = value
    else:
        aDict[key].append(value)

def plot_simple(ex_data,color_idx,mplot,startIdx,endIdx):
	line_type=lines[0]
	label_text=color_idx
	mplot.plot(ex_data['time'][startIdx:endIdx],ex_data['value'][startIdx:endIdx],line_type,linewidth=1,label=label_text,color=color_idx)
	line_type=next(linecycler)

def plot_best(ex_data,line_type,color_name,label_text,mplot):
	#label_texts.append(label)
	#line_type=next(linecycler)
	
	#size=len( ex_data['time']);
	#time = range (0, size*10,10);
	#temp_time = [ x/60.0 for x in time];
	#time = temp_time;
	#mplot.plot(ex_data['time'],ex_data['base'],line_type,linewidth=1, color='black' );
	mplot.plot(ex_data['time'],ex_data['value'],line_type,linewidth=2,label=label_text,color=color_name)
	line_type=next(linecycler)


plt.close();


# set directory and input *.csv file names
#Dir="/home/leo/Documents/forPrima2015/";
path_out="./"
out_name="learning_result.csv"
#Dir='/mnt/VrtualShared/WinShared/VirtShared/users/Lalith_ERI/Students/2014/Leo/ICEM2015_Taiwan/FullPaper/Figure/DataFromStphen/'

initial_path='/home/leo/Documents/PRIMA2016/'
initial_name="INITIAL_STATE.csv"

data_path = '../raw/'
data_path_out = '../output/'
target_name="user_"
start_num=500
best_val=0
best_name=""
#
#PREPARE PLOT
#

#use LaTex fonts
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig1 = plt.figure(figsize=(6,4))
sub1=fig1.add_subplot(1,1,1);
#axis intersect at
sub1.axhline(y=0, color='black');
sub1.axvline(x=0, color='black');
#remove top and right axes
sub1.spines['right'].set_visible(False)
sub1.spines['top'].set_visible(False)
sub1.yaxis.set_ticks_position('left')
sub1.xaxis.set_ticks_position('bottom')
sub1.hold(1)

#dictionary
#
#for filename in os.listdir(data_path):#
#	if target_name in filename:
#		print "OPENING ",filename#
#		fin = open(data_path+filename, 'r');
#		ex_data = np.genfromtxt(fin, delimiter=',', skip_header=1,skip_footer=0, names=['day','time', 'value']);
#		color_idx=int(ex_data['day'][0])		

startIdx=0;
endIdx=0;

count=0
for filename in os.listdir(data_path):
	#count+=1
	#if count >1:
	#	break
	if target_name in filename:
		
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

exit()

handles, labels = sub1.get_legend_handles_labels()

# or sort them by labels
#import operator
#hl = sorted(zip(handles, labels),
#            key=operator.itemgetter(1))
#handles2, labels2 = zip(*hl)
#
#sub1.legend(handles2, labels2, loc='upper left')

#exit()
#
#POST PROC PLOT
#
#plt.legend(label_texts, loc='upper left')

######################## plot ax1 ##########################
sub1.tick_params(axis='both', which='major', labelsize=14)
#sub1.set_ylim([0,20]); # y plot ranges
sub1.set_xlim([0,50]); # y plot ranges
#y_pos = np.arange(0,1,0.5);
#sub1.set_yticks(y_pos);
sub1.set_xlabel('Time (30min)', size=16);
sub1.set_ylabel("\n".join(textwrap.wrap('Raw Data',30)), size=16);


plt.draw();
plt.show();
plt.savefig(path_out+out_name+'.png',bbox_inches='tight', pad_inches=0);
plt.savefig(path_out+out_name+'.pdf',bbox_inches='tight', pad_inches=0);
