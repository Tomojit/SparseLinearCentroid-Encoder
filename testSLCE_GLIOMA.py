import pdb
import torch
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from utilityScript import *
from SparseLinearCentroidEncoderPyTorch import SCLCE
from simpleANNClassifierPyTorch import *

def load_GLIOMA_Data(dataSetName,partition):
	
	trnSet,tstSet = getApplicationData(dataSetName,partition)
	return trnSet,tstSet
	
def runSLCE(trData,trLabels,embeddingDim,lamda,learning_rate,cudaDeviceId):
	
	num_epochs = 2000
	standardizeFlag = False # data is already normalized so no need to do it again 
	miniBatch_size = trData.shape[1]
	model = SCLCE(embeddingDim,lamda)
	model.fit(trData,trLabels,standardizeFlag,learning_rate,miniBatch_size,num_epochs,cudaDeviceId,verbose=False)
	splWs = model.splWs.detach().to('cpu')
	feaList,feaW = returnImpFeaturesElbow(splWs)
	return feaList
	
def classifyGLIOMAData(D_tr,trLabels,D_tst,tstLabels,featureSet,fCntList,gpuId,pp):
	
	accuracyList = []
	
	for feaCnt in fCntList:
		
		#use the selected features
		fea = featureSet[:feaCnt]
		trData,tstData = D_tr[:,fea],D_tst[:,fea]	
		nClass = len(np.unique(trLabels))
		allACC = []
		for i in range(10):
			ann = NeuralNet(trData.shape[1], [500] , nClass)
			ann.fit(trData,trLabels,standardizeFlag=True,batchSize=64,optimizationFunc='Adam',learningRate=0.001, numEpochs=200,cudaDeviceId=gpuId)
			ann = ann.to('cpu')
			tstPredProb,tstPredLabel = ann.predict(tstData)
			accuracy = 100 * accuracy_score(tstLabels.flatten(), tstPredLabel)
			allACC.append(accuracy)	
		allACC = np.hstack((allACC))
		accuracyList.append(np.round(np.mean(allACC),2))
		
		print('Repetition:',pp+1,'Accuracy using',trData.shape[1],'of features:',np.round(np.mean(allACC),1))
	return accuracyList
	


if __name__ == "__main__":

	dataSet = 'GLIOMA'

	# initialize hyper-parameters for SLCE	
	embeddingDim = 5
	lamda = 0.30
	learning_rate = 0.002
	cudaId = 0
	fCntList = [10,50]

	topTenFeaturesAcc,topFiftyFeaturesAcc = [],[]
	for pp in range(20):

		#load training data data
		trnSet,tstSet = load_GLIOMA_Data('GLIOMA',pp)
		trData,trLabels = trnSet[:,:-1],trnSet[:,-1]
		tstData,tstLabels = tstSet[:,:-1],tstSet[:,-1]

		#data normalization
		muTr,sdTr,trData = standardizeData(trData)
		tstData = standardizeData(tstData,muTr,sdTr)
		
		#run SLCE on training data for feature selection
		feaList = runSLCE(trData,trLabels,embeddingDim,lamda,learning_rate,cudaId)

		#using the selected features run classification
		accuracyList = classifyGLIOMAData(trData,trLabels,tstData,tstLabels,feaList,fCntList,cudaId,pp)
		topTenFeaturesAcc.append(accuracyList[0])
		topFiftyFeaturesAcc.append(accuracyList[1])
	print('\t Mean accuracy using top 10 features over 20 run',np.round(np.mean(topTenFeaturesAcc),1))
	print('\t Mean accuracy using top 50 features over 20 run',np.round(np.mean(topFiftyFeaturesAcc),1))

