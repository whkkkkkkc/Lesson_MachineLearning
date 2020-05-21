'''
@ brief	:	BP network for Machine Learning
@ Author:	Vayhoon Yan
@ Date	:	2020/5/19
@Version:	1.0
'''
####################################################
# import neccesary lib
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split


'''

'''
class BPNetwork(object):
	'''
	@ brief: initialize the weight and bias as the object of parameters list
	@ argv: W: random number for initial
	        B: zeros
	'''
	def __init__(self, InputSize, HideSize, OutputSize, precision = 1e-6):
		self.ParaDic = {}
		# connection between input and hide Net
		self.ParaDic['W1'] = precision * np.random.randn(InputSize,HideSize)
		self.ParaDic['B1'] = np.zeros(HideSize)
		# connection between hide and output Net 
		self.ParaDic['W2'] = precision * np.random.randn(HideSize,OutputSize)
		self.ParaDic['B2'] = np.zeros(OutputSize)
	'''
	@ brief: calculate the forward output
	@ argv: 	X: input
	'''
	def Forward(self,X):
		# forward propagation
			# calculate the hide net
		Hide = np.maximum(np.dot(X,self.ParaDic['W1']) + self.ParaDic['B1'],0)
			# calculate the output 
		Ycal = np.dot(Hide,self.ParaDic['W2']) + self.ParaDic['B2']
		return Hide,Ycal
	'''
	@ brief: backward process to calculate grad
	@ argv : 	Yref: reference output
			  Xtrain: train dataset
				Hide: the output of Hide layer
				Ycal: forward propogation output
				   N: batch size of the train dataset
			  lambda: coefficient
	'''
	def CalGrad(self,Xtrain,Yref,Hide,Ycal,N,lamda):
		grad = {}
		# backward of softmax
		Alpha = np.exp(Ycal)/((np.exp(Ycal)).sum(axis=1,keepdims=True))
		Alpha[range(N),Yref] -= 1 	# renew the index
		Alpha = Alpha/N 			#
		# get the hide net grad to renew the wight and bias 
		grad['B2'] = np.sum(Alpha,axis=0)
		grad['W2'] = np.dot(Hide.T,Alpha) + lamda*self.ParaDic['W2']
		# backward of ReLU
		Dhide = np.dot(Alpha, self.ParaDic['W2'].T)
		Dhide[(np.dot(Xtrain,self.ParaDic['W1'])+self.ParaDic['B1'])<0] = 0
		grad['B1'] = np.sum(Dhide,axis=0)
		grad['W1'] = np.dot(Xtrain.T,Dhide) + lamda*self.ParaDic['W1']
		#print("grad result:\n", grad)
		return grad
	'''
	@ brief: calculate the loss via Softmax function
	@ argv:		Yref: reference output
				Ycal: forward propogation output
				   N: batch size of the train dataset
			  lambda: coefficient
	'''
	def Softmax(self,Yref,Ycal,N,lamda):
		loss = None
		loss = -Ycal[range(N),Yref].sum() + np.log(np.exp(Ycal).sum(axis = 1)).sum()
		loss = loss/N + 0.5*lamda*(np.sum(self.ParaDic['W1']*self.ParaDic['W1'])+np.sum(self.ParaDic['W2']*self.ParaDic['W2']))
		return loss
	'''
	@ brief: training process
	@ argv:		Xtrain: training dataset
				Ytrain: training reference
				Xval  : validation dataset
				Yval  : validation reference
				LearnRate: Learn rate
				RateDecay: decay coefficient of learn rate
				lamda : regularization coefficient
				IterationInterval: Max interation times
				BatchSize: divide training set into batches
				echo : flag to print the loss information 
	'''
	def train(self, Xtrain,Ytrain,Xval,Yval,
		LearnRate=1e-2,RateDecay=0.9,lamda=5e-6,IterationInterval=10000,BatchSize=200,echo=True):
		TrainNum = Xtrain.shape[0]
		# save information during training
		LossInfo = []
		TrainAcuracyInfo = []
		ValAcuracyInfo = []
		for iteration in range(IterationInterval):
			Xbatch = None
			Ybatch = None
			batch = np.random.choice(TrainNum,BatchSize,replace=True)
			Xbatch = Xtrain[batch]
			Ybatch = Ytrain[batch]
			# process of training
			Hide,Ycal = self.Forward(Xbatch)
			loss = self.Softmax(Ybatch,Ycal,BatchSize,lamda)
			grad = self.CalGrad(Xbatch,Ybatch,Hide,Ycal,BatchSize,lamda)
			# collect loss information
			LossInfo.append(loss)
			# renew the weight and bias
			self.ParaDic['W1'] -= LearnRate * grad['W1']
			self.ParaDic['B1'] -= LearnRate * grad['B1']
			self.ParaDic['W2'] -= LearnRate * grad['W2']
			self.ParaDic['B2'] -= LearnRate * grad['B2']
			# echo board, it epoch==True then will print the loss information of each iteration 
			if echo and iteration%100 == 0:
				print("iteration %d/%d: loss = %f" % (iteration,IterationInterval,loss))
			if iteration%50 == 0:
				# collect information each epoch
				TrainAcuracy = (self.test(Xbatch)==Ybatch).mean()
				ValAcuracy = (self.test(Xval)==Yval).mean()
				TrainAcuracyInfo.append(TrainAcuracy)
				ValAcuracyInfo.append(ValAcuracy)
				# decay the learning rate each epoch
				#LearnRate *= RateDecay
		return {	"training accuracy information": TrainAcuracyInfo,
					"validation accuracy information": ValAcuracyInfo,
					"loss information": LossInfo,
		}
	def test(self,Xtest):
		Ytest = None
		Hide = np.maximum(np.dot(Xtest,self.ParaDic['W1'])+self.ParaDic['B1'],0)
		Ytest= np.argmax(np.dot(Hide,self.ParaDic['W2'])+self.ParaDic['B2'],axis=1)
		return Ytest
	def predict(self,Xpredict,Ypredict):
		accuracy = (self.test(Xpredict)==Ypredict).mean()
		return accuracy
'''
@ brief: prepare dataset using sklearn built-in function
'''
def PrepareDataset():
	# load digits dataset from sklearn
	digits = datasets.load_digits()
	images = digits.images
	reference = digits.target
	#print("image scale: ",images.shape)
	# divide the dataset into train and test through the function train_test_split()
	Dtrain,Xtest,YtrainD,Ytest = train_test_split(images,reference,test_size=0.2,random_state=0)
	Xtrain,Xval,Ytrain,Yval = train_test_split(Dtrain,YtrainD,test_size=0.2,random_state=0)
	TrainNum = Xtrain.shape[0]
	ValNum = Xval.shape[0]
	TestNum = Xtest.shape[0]
	#print("Xtrain object numbers:\n ", TrainNum)
	#print("Xval object numbers:\n ", ValNum)
	#print("Xtest object numbers:\n ", TestNum)
	# change the 8*8 image data into 64*1 image data
	Xtrain = Xtrain.reshape(TrainNum,-1)
	Xval = Xval.reshape(ValNum,-1)
	Xtest = Xtest.reshape(TestNum,-1)
	#print("training dataset shape: ", Xtrain.shape)
	#print("validation dataset shape: ", Xval.shape)
	#print("test dataset shape: ", Xtest.shape)
	return Xtrain,Ytrain,Xval,Yval,Xtest,Ytest

def DrawScatter(N,Y):
	X = np.linspace(0,N,N)
	ax = plt.figure().add_subplot(111)
	ax.set_title('loss curve')
	plt.xlabel('Iteration')
	plt.ylabel('Loss')
	ax.scatter(X,Y,c='r',marker='o')
	plt.show()
'''
@ brirf: instantation a BPNetwork object, then run the train  
'''
def runBpNetwork(HideSize=40):
	Xtrain,Ytrain,Xval,Yval,Xtest,Ytest=PrepareDataset()
	InputSize = Xtrain.shape[1]
	OutputSize= 10
	IterationInterval=4000
	BPnet = BPNetwork(InputSize,HideSize,OutputSize)
	start = time.time()
	TrainInfo = BPnet.train(Xtrain,Ytrain,Xval,Yval,IterationInterval=IterationInterval,BatchSize=200,echo=True,LearnRate=0.1)
	end = time.time()
	print("training time: ",end - start)
	print("accuracy for validation dataset: ", BPnet.predict(Xval,Yval))
	print("accuracy for test dataset: ", BPnet.predict(Xtest,Ytest))
	LossArray=np.array(TrainInfo['loss information'])
	DrawScatter(IterationInterval,LossArray)
	#print(TrainInfo)
	#TestInfo = BPnet.test()
'''
'''
#PrepareDataset()
runBpNetwork()