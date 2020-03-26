'''
module: 	LinearRegression.py
version:	1.0
author:		Yan Weihong
date:		2020 Mar.
'''
##

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import csv

LinearData = [[2104,3,400],[1600,3,330],[2400,3,369],[1416,2,232],[3000,4,540]]
LinearData = np.array(LinearData)
InputData = [[1,1,1],[2005,3200,1280],[3,4,2]]
##print(LinearData)
##
def PlotScatter(LinearData):
	X = LinearData[:,-3].astype('float32')
	Y = LinearData[:,-2].astype('float32')
	Z = LinearData[:,-1].astype('float32')
	fig = plt.figure()
	ax=plt.subplot(111,projection='3d')
	ax.scatter(X,Y,Z)
	ax.set_zlabel('price')
	ax.set_ylabel('bedrooms')
	ax.set_xlabel('area')

	x = np.arange(1300,3300,100)
	y = np.arange(1,5,0.2)
	z = -70.43460148 + x*0.0638433756 + y*103.436047
	ax.plot(x,y,z,label='predict')

	plt.show()
	return 0

#PlotScatter(LinearData)

def TheNormalEquation(LinearData):
	## data preparation
	X = np.ones((5,3))
	X[:,-2] = LinearData[:,-3]
	X[:,-1] = LinearData[:,-2]
	Y = np.ones((5,1))
	Y = LinearData[:,-1]
	XT = np.mat(X).transpose()
	XTX = np.dot(XT,X)
	## Inverse matrix using built-in function
	XTX_1 = np.linalg.inv(XTX)
	## solve the value of theta
	theta = np.dot(XTX_1,XT)
	theta = np.dot(theta,Y)
	print(theta)
	return theta
	## 
def test(InputData,theta):
	#theta = np.mat(theta).transpose()
	Y = np.dot(theta,InputData)
	print(Y)
	return Y
'''
## The normal Equation
theta = TheNormalEquation(LinearData)
print(theta)
test(InputData,theta)
'''
def GradientDescent():
	'''
	need to insert comments
	need to pack the data preparation process
	'''
	mean_X = 2104
	std_X = 568.821940505111
	X = [[1.0,0.0,0.0],[1.0,-0.886041772,0.0],[1.0,0.520373739,0.0],[1.0,-1.209517339,-1.58113883],[1.0,1.575185372,1.58113883]]
	Y = [400,330,369,232,540]
	MaxIteration = 50000
	delta = 0.0001
	TrainData = np.mat(X)
	print(TrainData)
	TrainLabel= np.mat(Y).transpose()
	theta = np.ones((3,1))
	cost = np.ones((50000,3))
	for i in range(MaxIteration):
		H_theta = np.dot(TrainData,theta)
		loss = H_theta - TrainLabel
		tmp = np.dot(TrainData.transpose(),loss)
		cost[i,:] = tmp.transpose()
		if tmp.all()<0.1:
			print(tmp)
			break
		theta = theta - delta*tmp
	cost_1 = abs(cost[:,-3])
	cost_2 = abs(cost[:,-2])
	cost_3 = abs(cost[:,-1])
	np.savetxt('cost1.txt',cost_1)
	np.savetxt('cost2.txt',cost_2)
	np.savetxt('cost3.txt',cost_3)
	print(theta)
	In = [[1.0,-0.174043919,0.0],[1.0,1.92678925,1.58113883],[1.0,-1.448607976,-1.58113883]]
	In = np.mat(In)
	print(np.dot(In,theta))