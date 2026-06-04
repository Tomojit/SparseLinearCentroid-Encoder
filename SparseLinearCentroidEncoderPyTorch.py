'''
Name:Tomojit Ghosh
Program Name: SparseLinearCentroidEncoder.py
Purpose: Solving the LCE optimization with sparsity on input features.
'''

import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
from OneToOneLinear import OneToOneLinear
import copy
#torch.set_default_dtype(torch.float64)
#torch.set_printoptions(precision=64)
class SCLCE(nn.Module):

	def __init__(self,embeddingDim=2,penalty=0.1):
		super(SCLCE, self).__init__()
		self.embeddingDim = embeddingDim
		self.penalty = penalty
		self.valErrorTrace = []
		self.trMu = []
		self.initW = []
		self.errorTraceWOSparsity = []
		self.errorTraceWSparsity = []
		self.l1RegTrace = []
		self.l2RegTrace = []
		self.regTrace = []
		self.maxEpochs = 5000
		self.preTrEpochs = 10
		

	def initWeight(self):
		
		#set the lower and upper bound of distribution and randomly initialize the matrix A
		r1 = -1/np.sqrt(self.inputDim)
		r2 = 1/np.sqrt(self.inputDim)
		data = (r2 - r1) * torch.rand(self.inputDim, self.embeddingDim) + r1
		self.W = nn.Parameter(data)
		#now initialize the SPL layer
		self.splWs = nn.Parameter(torch.ones(self.inputDim,))
		
	def createOutputAsCentroids(self,data,label):
		centroidLabels=np.unique(label)
		outputData=np.zeros([np.shape(data)[0],np.shape(data)[1]])
		for i in range(len(centroidLabels)):
			indices=np.where(centroidLabels[i]==label)[0]
			tmpData=data[indices,:]
			centroid=np.mean(tmpData,axis=0)
			outputData[indices,]=centroid
		return outputData
		
	def reconstruction(self,X,sparseFlag):
		#pdb.set_trace()
		if sparseFlag:
			#pass the data through the SPL layer, approach 1
			D = torch.diag(self.splWs)
			X = torch.matmul(X,D)

		#build the matrix A to reconstruct data
		A = torch.matmul(self.W,torch.t(self.W))
					
		return torch.matmul(X,A)
		

	def train(self,dataLoader,learningRate,miniBatchSize,numEpochs,verbose):

		def costFuncWOSparsity(X,C):
			#pdb.set_trace()
			nSample = X.shape[0]
			Q = C - self.reconstruction(X,False)
			cost = 0.5*(1/nSample)*torch.trace(torch.matmul(torch.t(Q),Q))
			return cost
			
		def costFuncWSparsity(X,C):
			#pdb.set_trace()
			nSample = X.shape[0]		
			Q = C - self.reconstruction(X,True)
			cost = 0.5*(1/nSample)*torch.trace(torch.matmul(torch.t(Q),Q))
			
			#calculate the L1 loss
			if self.penalty == None:
				sparsityLoss = 0
			else:
				sparsityLoss = self.penalty*torch.norm(self.splWs, p=1)
			cost = cost + sparsityLoss
			return cost,sparsityLoss

		# Load the model to device
		self.to(self.device)
		
		# set optimization function		
		optimizer = torch.optim.Adam(self.parameters(),lr=learningRate,amsgrad=True)

		#first solve the problem ||C - AA.T X||^2_F by updating A 
		#training with gradient descent
		keepTraining = True
		epoch = 0
		while keepTraining:
			error = []
			for i, (X,Y) in enumerate(dataLoader):  
				# Move tensors to the configured device
				X = X.to(self.device)
				Y = Y.to(self.device)
				# calculate cost
				#pdb.set_trace()
				loss = costFuncWOSparsity(X,Y)
				error.append(loss.item())
				
				# calculate gradient and update parameters
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			epoch += 1
			self.errorTraceWOSparsity.append(torch.mean(torch.FloatTensor(error)))
			#check for terminating criteria
			if epoch > self.maxEpochs:
				keepTraining = False
				#print('Max. epochs reached. Stopping training.')
			elif epoch >1 and torch.abs(self.errorTraceWOSparsity[-1] - self.errorTraceWOSparsity[-2]) <= 0.000001:
				keepTraining = False
				self.optimalEpoch = epoch
				#print('Stopping criteria reached. Optimal no of epoch:',self.optimalEpoch)
			#print epoch error
			if verbose and ((epoch) % 25) == 0:
				print ('Epoch: {}, Loss: {:.10f}'.format(epoch, self.errorTraceWOSparsity[-1]))

		#now apply sparsity while keeping the weights on A fixed
		self.W.requires_grad = False
		#first adjust the weights of sparse layer without the penalty
		tmpPenalty = self.penalty
		self.penalty = None
		self.sparsityType = None
		for epoch in range(self.preTrEpochs):
			error = []
			for i, (X,Y) in enumerate(dataLoader):  
				# Move tensors to the configured device
				X = X.to(self.device)
				Y = Y.to(self.device)
				# calculate cost
				loss,_ = costFuncWSparsity(X,Y)
				error.append(loss.item())
				
				# calculate gradient and update parameters
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			#print epoch error
			if verbose and ((epoch+1) % (numEpochs*0.1)) == 0:
				print ('Epoch [{}/{}], Loss: {:.10f}'.format(epoch+1, self.preTrEpochs, torch.mean(torch.FloatTensor(error))))

		#pdb.set_trace()
		self.penalty = tmpPenalty
		
		#now train with 1-norm applied to the sparsity promoting layer
		for epoch in range(numEpochs):
			error = []
			sparsityLoss = []
			for i, (X,Y) in enumerate(dataLoader):  
				# Move tensors to the configured device
				X = X.to(self.device)
				Y = Y.to(self.device)
				# calculate cost
				loss,regLoss = costFuncWSparsity(X,Y)
				error.append(loss.item())
				sparsityLoss.append(regLoss.item())
				error.append(loss.item())
				# calculate gradient and update parameters
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			self.errorTraceWSparsity.append(torch.mean(torch.FloatTensor(error)))
			self.regTrace.append(torch.mean(torch.FloatTensor(sparsityLoss)))
			#print epoch error
			if verbose and ((epoch+1) % (numEpochs*0.1)) == 0:
				print ('Epoch [{}/{}], Loss: {:.10f}, L1 Sparsity loss: {:.10f}'.format(epoch+1,numEpochs,self.errorTraceWSparsity[-1], self.regTrace[-1]))
				

	def fit(self,X,L,standardizeFlag=True,learningRate=0.001, miniBatchSize=100,numEpochs=100,cudaDeviceId=0,verbose=True):

		# set device
		self.device = torch.device('cuda:'+str(cudaDeviceId))
		
		#standardize data
		if standardizeFlag:
			self.trMu = np.mean(X,axis=0)
			X = X - self.trMu
			self.trMu = torch.from_numpy(self.trMu).float().to(self.device)

		#create target: centroid for each class
		C = self.createOutputAsCentroids(X,L)

		#hold the training sample count
		X = torch.from_numpy(X).float()
		C = torch.from_numpy(C).float()
		#X = torch.from_numpy(X)
		#C = torch.from_numpy(C)
		self.inputDim = X.shape[1]
		self.trDataSize = X.shape[0]

		#pdb.set_trace()
			
		#Prepare data for torch
		trDataTorch = Data.TensorDataset(X,C)
		dataLoader = Data.DataLoader(dataset=trDataTorch,batch_size=miniBatchSize,shuffle=True)
		
		#initialize the network weight
		self.initWeight()
		self.inputDim = torch.tensor(X.shape[1],dtype=torch.float32).cuda()
		#pdb.set_trace()

		#training
		self.train(dataLoader,learningRate,miniBatchSize,numEpochs,verbose)
		
		#self.splWs = self.splWs.detach().to('cpu')
		self.splWs.detach().to('cpu')
		self.fWeight,self.fIndices = torch.sort(torch.abs(self.splWs),descending=True)
		
	def transform(self,Y):
		# Method to transform data into low dimensional space 
		#pdb.set_trace()
		Y = torch.from_numpy(Y).float().to(self.device)
		if len(self.trMu) != 0: #substract training mean from test data
			Y = Y - self.trMu
		with torch.no_grad():#we don't need to compute gradients (for memory efficiency)
			#Y1 = self.Layer[0](Y)
			#Z = torch.t(torch.matmul(self.Layer[1].weight,torch.t(Y1)))
			Y1 = torch.matmul(torch.diag(self.W_spl),Y)
			Z = torch.matmul(self.W,torch.t(Y1))
		Z = Z.to('cpu')
		return Z
		
	def predict(self,Y):
		# method to predict the class label of the test samples
		Y = torch.from_numpy(Y).float().to(self.device)
		if len(self.trMu) != 0: #substract training mean from test data
			Y = Y - self.trMu
		with torch.no_grad():#we don't need to compute gradients (for memory efficiency)
			#Y1 = self.Layer[0](Y)
			#Z = torch.t(torch.matmul(self.Layer[1].weight,torch.t(Y1)))
			Y1 = torch.matmul(torch.diag(self.W_spl),Y)
			Z = torch.matmul(self.W,torch.t(Y1))
		Z = Z.to('cpu')

