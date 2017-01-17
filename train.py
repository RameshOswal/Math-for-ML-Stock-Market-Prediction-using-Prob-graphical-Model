# Dee Dong Explained me the process of how to approach the problem.
import numpy as np
import math

def featureExtractionTrain(data):
    features = np.zeros([len(data), 5 * len(data[0])])
    for col in range(len(data[0])):
        for j in range(5, len(data)):
            for k in range(4, -1, -1):
                features[j, 5 - k - 1 + (5 * col)] = math.log(data[j - k, col] / data[j - k - 1, col])
    return(features)


def covariance(feat):
    #covMatrix  = np.zeros([len(feat[0]),len(feat[0])])
    covMatrix =  np.matmul(feat.T, feat)
    covMatrix = 0.0025316 *covMatrix
    print(covMatrix)
    return covMatrix


def __main__():
    #feature extraction for train data
    data = np.genfromtxt('train.csv',delimiter=',')
    data = data[1:,:]
    featuresTrain = featureExtractionTrain(data)
    featuresTrain = featuresTrain[5:,:20]
    #np.savetxt('featuresTrain.csv',featuresTrain,delimiter=',')
    #print(featuresTrain)

    #calculate mean_u and mean_v on the trainning set
    meanTrain = np.mean(featuresTrain,axis=0)
    np.savetxt('mean.csv',meanTrain,delimiter=',')
    #Covariance matrix for train data
    covarianceMatrix = covariance(featuresTrain)
    np.savetxt('covarianceMatrix.csv', covarianceMatrix, delimiter=',')
    np.savetxt('precision.csv',np.linalg.inv(covarianceMatrix),delimiter=',')
__main__()

