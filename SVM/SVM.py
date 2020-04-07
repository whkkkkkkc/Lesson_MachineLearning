'''
module: 	SVM.py
version:	1.0
author:		Yan Weihong
date:		2020 Apr.
'''
##
import 	numpy as np
import	os
import	random
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn import svm

def load_jpg(IMG_PATH):
	'''
	load a picture as numpy.ndarray
	'''
	if IMG_PATH < 10:
		IMG_PATH = '0' + str(IMG_PATH)
	else:
		IMG_PATH = str(IMG_PATH)
	image = Image.open(IMG_PATH+'.jpg')
	R_img,G_img,B_img = image.split()
	R_array = np.array(R_img)
	G_array = np.array(G_img)
	B_array = np.array(B_img)
	#print(R_array)
	R = np.mean(R_array)
	G = np.mean(G_array)
	B = np.mean(B_array)
	R_var = np.var(R_array)
	G_var = np.var(G_array)
	B_var = np.var(B_array)
	R_std = np.std(R_array,ddof=1)
	G_std = np.std(G_array,ddof=1)
	B_std = np.std(B_array,ddof=1)
	IMG_mean = (R+G+B)/3
	IMG_var = (R_var+G_var+B_var)/3
	IMG_std = (R_std+G_std+B_std)/3
	return IMG_mean,IMG_var,IMG_std

def load_bmp(BMP_PATH):
	BMP_PATH = str(BMP_PATH)
	image = Image.open(BMP_PATH+'.bmp')
	Array = np.array(image)
	BMP_mean = np.mean(Array)
	BMP_var = np.var(Array)
	BMP_std = np.std(Array,ddof=1)
	return BMP_mean,BMP_var,BMP_std

def pre_data(Ratio):
	TestDataArray = []
	TestLabel = []
	TrainDataArray = []
	TrainLabel = []
	type_0 = []
	type_1 = []
	for i in range(1,22):
		if i < 13:
			i_mean,i_var,i_std = load_jpg(i)
			label = 0.0
			type_0.append([i_mean,i_var,i_std])
		else:
			i_mean,i_var,i_std = load_bmp(i)
			label = 1.0
			type_1.append([i_mean,i_var,i_std])
		if random.randint(0,99)>Ratio:
			TestDataArray.append([i_mean,i_var,i_std])
			TestLabel.append(label)
		else:
			TrainDataArray.append([i_mean,i_var,i_std])
			TrainLabel.append(label)
	return type_0,type_1,TrainDataArray,TrainLabel,TestDataArray,TestLabel
'''
plot function
'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def plot_scatter_line(type_total,type_0,type_1,coef_line):
	'''
	type_total denotes the total data
	type_0 denotes the type 0 data
	type_1 denotes the type 1 data
	coef_line denotes the linear SVM result

	'''
	ax = plt.figure().add_subplot(111, projection = '3d')
	## normalization process
	ss2 = StandardScaler()
	total = ss2.fit_transform(type_total)
	type_0 = ss2.transform(type_0)
	type_1 = ss2.transform(type_1)
	## scatter plot process
	ax.scatter(type_0[:,-3],type_0[:,-2],type_0[:,-1],c='r',marker='^')
	ax.scatter(type_1[:,-3],type_1[:,-2],type_1[:,-1],c='g',marker='o')
	ax.set_xlabel('MEAN')
	ax.set_ylabel('VAR')
	ax.set_zlabel('STD')
	## line plot process
	xline = np.linspace(-2,2,100)
	yline = np.linspace(-2,2,100)
	zline = (xline*coef_line[0,-3]+yline*coef_line[0,-2])/(-coef_line[0,-1])
	ax.plot(xline,yline,zline,label="SVM")
	## show the final image
	plt.show()


def LinearSvm(Ratio,l):
	type_0,type_1,TrainDataArray,TrainLabel,TestDataArray,TestLabel=pre_data(Ratio)
	type_0 = np.array(type_0)
	type_1 = np.array(type_1)
	TrainDataArray = np.array(TrainDataArray)
	TestDataArray = np.array(TestDataArray)
	Ytrain = np.array(TrainLabel)
	Ytest = np.array(TestLabel)
	## normalize process
	ss = StandardScaler()
	Xtrain = ss.fit_transform(TrainDataArray)
	Xtest = ss.transform(TestDataArray)
	## SVM process
	#clf = svm.SVC(C=0,kernel='linear')			## linear kernel
	#clf = svm.SVC(C=l,kernel='linear')
	clf = svm.SVC(kernel='rbf',gamma=1)	## Gauss kernel
	#clf = svm.SVC(kernel='poly')			## polynomial kernel
	## SVM training using built-in function and class
	clf.fit(Xtrain,Ytrain)		## similar with train process
	result = clf.predict(Xtest) ## predict process
	print("SVM result:",result)				## predict result
	print("test reference:",Ytest)				## comparasion
	## work only linear kernel
	#print("linear corfficients:",clf.coef_)
	#coef_line = clf.coef_
	#plot_scatter_line(TrainDataArray,type_0,type_1,coef_line)

LinearSvm(90,0.1)
LinearSvm(90,0.1)
LinearSvm(90,0.1)
LinearSvm(90,0.1)
LinearSvm(90,0.1)
LinearSvm(90,0.1)
LinearSvm(90,0.1)
LinearSvm(90,0.1)
LinearSvm(90,0.1)
