'''
module: 	cluster.py
version:	1.0
author:		Yan Weihong
date:		2020 Apr.
'''

##
import 	numpy as np
import	os
import	random
import	matplotlib.pyplot as plt
import math

##
data = [[0,0],[1,0],[0,1],[1,1],[2,1],[1,2],[2,2],[3,2],[6,6],[7,6],
		[8,6],[6,7],[7,7],[8,7],[9,8],[7,8],[8,8],[9,9],[8,9],[9,5]]
data=np.array(data)

def plot_scatter(data):
	ax = plt.figure().add_subplot(111)
	ax.set_title('Scatter Plot')
	plt.xlabel('Feature 1')
	plt.ylabel('Feature 2')
	print(data)
	'''k=2 plot
	#ax.scatter(x=data[0:7,0],y=data[0:7,1],c='g',marker='o')
	#ax.scatter(x=data[8:19,0],y=data[8:19,1],c='b',marker='^')
	'''
	'''k=3 plot
	ax.scatter(x=data[0:7,0],y=data[0:7,1],c='r',marker='^')
	ax.scatter(x=data[8:12,0],y=data[8:12,1],c='g',marker='o')
	ax.scatter(x=data[19,0],y=data[19,1],c='g',marker='o')
	ax.scatter(x=data[13:18,0],y=data[13:18,1],c='b',marker='o')
	'''
	#ax.scatter(x=data[:,0],y=data[:,1],c='b',marker='^')

	ax.scatter(x=data[0:3,0],y=data[0:3,1],c='r',marker='^')
	ax.scatter(x=data[3:8,0],y=data[3:8,1],c='g',marker='^')
	ax.scatter(x=data[8:13,0],y=data[8:13,1],c='b',marker='o')
	ax.scatter(x=data[19,0],y=data[19,1],c='b',marker='o')
	ax.scatter(x=data[13:19,0],y=data[13:19,1],c='y',marker='o')
	
	plt.show()

def Kmeans(data,k):
## data
## k means data will be divide into k groups	
	centers = gen_start(data,k,3)
	print('centers of cluster:\n',centers)
	label = gen_cluster(data,centers,k)
	return label

## calculate distance
def d_Manhattan(data_1,data_2):
	return abs(data_1[0]-data_2[0])+abs(data_1[1]-data_2[1])
	

def gen_start(data,k,min_d=3):
	centers = []
	centers.append(data[random.randint(0,19)]) 	## generate the first random start point
	for i in range(1,k):						## generate the other k-1 start point 
		flag = 0								
		while flag==0:
			j = random.randint(0,19)
			print('generate random point:',j)
			flag_l = 1
			for l in range(0,i):
				# print('compare data: ',data[centers[l]],data[j])
				# print('manhattan distance: ',d_Manhattan(centers[l],data[j]))
				if d_Manhattan(centers[l],data[j])<min_d:
					flag_l=0
					break
			flag = flag_l
		centers.append(data[j])
	centers = np.array(centers)
	return centers
##

## repeat generate cluster process
def gen_cluster(data,centers,k):
	label_new = np.zeros((20,1))
	center_i = np.zeros((k,3))
	iteration = 0
	while (True):
		i = 0
		while (i<20):
			min_d = 100
			min_i = 100
			j = 0
			while (j<k):
				if min_d>d_Manhattan(data[i],centers[j]):
					min_d = d_Manhattan(data[i],centers[j])
					min_i = j
				j += 1
			label_new[i]= min_i
			center_i[min_i,0] += data[i,0]
			center_i[min_i,1] += data[i,1]
			center_i[min_i,2] += 1
			i += 1
			## calculate the distance between the centers and each input point
			## accumulate the point data according to the label
		if(iteration>1000):
			break
		else:
			l = 0
			while(l<k):
				centers[l,0] = center_i[l,0]/center_i[l,2]
				centers[l,1] = center_i[l,1]/center_i[l,2]
				l += 1
				## calculate the new centers
		iteration += 1
		## count the iteration times
	return label_new

print('labels:\n', Kmeans(data,4))
plot_scatter(data)

'''
plot_scatter(data)
'''
