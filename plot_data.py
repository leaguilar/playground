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

def plot_simple(ex_data,color_idx,label_text,mplot,startIdx,endIdx):
	#label_texts.append(label)
	#line_type=next(linecycler)
	line_type=lines[0]
	size=len( ex_data['time']);
	time = range (0, size*10,10);
	temp_time = [ x/60.0 for x in time];
	time = temp_time;
	#mplot.plot(ex_data['time'],ex_data['base'],line_type,linewidth=1, color='black' );
	mplot.plot(ex_data['time'][startIdx:endIdx],ex_data['value'][startIdx:endIdx],line_type,linewidth=1,label=label_text,color=colors[color_idx])
	line_type=next(linecycler)

def plot_best(ex_data,line_type,color_name,label_text,mplot):
	#label_texts.append(label)
	#line_type=next(linecycler)
	
	size=len( ex_data['time']);
	time = range (0, size*10,10);
	temp_time = [ x/60.0 for x in time];
	time = temp_time;
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
data_path2 = '../output/'
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
	count+=1
	if count >1:
		break
	if target_name in filename:
		
		fin = open(data_path+filename, 'r');
		ex_data = np.genfromtxt(fin, delimiter=',', skip_header=0,skip_footer=0, names=['day','time', 'value']);
		color_idx=int(ex_data['day'][0])		
		fin.close();
		#plot_best(ex_data,"-","gray","",sub1)
		print "OPENING ",filename,int(ex_data['day'][0])
		startIdx=0;
		endIdx=0;
		color_idx=0
		for i in xrange(1,ex_data['day'].size):
			if ex_data['day'][i-1]!=ex_data['day'][i]:
				endIdx=i-1
				if int(ex_data['day'][startIdx]) <= 240 and int(ex_data['day'][startIdx]) >= 240:
					print startIdx,endIdx,ex_data['time'][startIdx],ex_data['time'][endIdx]
					plot_simple(ex_data,color_idx,color_idx,sub1,startIdx,endIdx)
				startIdx=endIdx+1				
			
				#plot_simple(ex_data,color_idx,color_idx,sub1,startIdx,endIdx)

		#if int(ex_data['day'][0]) < 240 and int(ex_data['day'][0]) > 213:
		#	plot_simple(ex_data,color_idx,color_idx,sub1)
		#print "FINISHED ",filename
		#exit()

handles, labels = sub1.get_legend_handles_labels()

##########################
########################
######################

startIdx=0;
endIdx=0;

count=0
for filename in os.listdir(data_path2):
	count+=1
	if count >1:
		break
	if target_name in filename:
		
		fin = open(data_path2+filename, 'r');
		ex_data = np.genfromtxt(fin, delimiter=',', skip_header=0,skip_footer=0, names=['day','time', 'value']);
		color_idx=int(ex_data['day'][0])		
		fin.close();
		#plot_best(ex_data,"-","gray","",sub1)
		print "OPENING ",filename,int(ex_data['day'][0])
		startIdx=0;
		endIdx=0;
		color_idx=0
		for i in xrange(1,ex_data['day'].size):
			if ex_data['day'][i-1]!=ex_data['day'][i]:
				endIdx=i-1
				if int(ex_data['day'][startIdx]) <= 240 and int(ex_data['day'][startIdx]) >= 240:
					print startIdx,endIdx,ex_data['time'][startIdx],ex_data['time'][endIdx]
					plot_simple(ex_data,color_idx,color_idx,sub1,startIdx,endIdx)
				startIdx=endIdx+1				
			
				#plot_simple(ex_data,color_idx,color_idx,sub1,startIdx,endIdx)

		#if int(ex_data['day'][0]) < 240 and int(ex_data['day'][0]) > 213:
		#	plot_simple(ex_data,color_idx,color_idx,sub1)
		#print "FINISHED ",filename
		#exit()



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
y_pos = np.arange(0,20,10);
sub1.set_yticks(y_pos);
sub1.set_xlabel('Time (30min)', size=16);
sub1.set_ylabel("\n".join(textwrap.wrap('Raw Data',30)), size=16);


plt.draw();
#plt.show();
plt.savefig(path_out+out_name+'.png',bbox_inches='tight', pad_inches=0);
plt.savefig(path_out+out_name+'.pdf',bbox_inches='tight', pad_inches=0);
