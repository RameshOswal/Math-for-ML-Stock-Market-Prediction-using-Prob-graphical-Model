# Dee Dong Explained me the process of how to approach the problem.
import numpy as np

import math

def featureExtractionVal(data):
    features = np.zeros([len(data), 5 * len(data[0])])
    features[:,:] = np.nan
    for col in range(len(data[0])):
        for j in range(len(data)):
            for k in range(4, -1, -1):
                if ((j - k)>=0 and  (col)>=0 and    (j - k - 1)>=0 and (col)>=0):
                    features[j, 5 - k - 1 + (5 * col)] = math.log(data[j - k, col] / data[j - k - 1, col])
    return (features)

def marg(matrix,indexes):
    resCov = np.zeros([len(indexes),len(indexes)])
    for row,ind1 in enumerate(indexes):
        for col,ind2 in enumerate(indexes):
            resCov[row,col] = matrix[ind1,ind2]
    return  resCov


def findNonNaN(f):
    nanIndex = np.isnan(f)
    index = []
    for i,val in enumerate(nanIndex):
        if(val == False):
            index.append(i)
    #print(index)
    return index
def __main__():
    #calculate mean_u and mean_v on the trainning set
    meanTrain = np.genfromtxt('mean.csv',delimiter=',')
    covarianceMatrix = np.genfromtxt('covarianceMatrix.csv',delimiter=',')

    #feature extraction for validation dataset
    data = np.genfromtxt('val.csv', delimiter=',')
    data = data[1:, :]
    featuresVal = featureExtractionVal(data)
    #print(featuresVal)
    featuresVal = featuresVal[:,:20]
    #np.savetxt('featuresVal.csv', featuresVal, delimiter=',')
    #print(covarianceMatrix)
    #predict values for validation to check model accuracy
    prediction = np.zeros(len(featuresVal))

    for i in range(1,len(featuresVal)):
        #value to be predicted
        featPredict = 4
        #find row indexes which we know
        nonNaNIndex = np.array(findNonNaN(featuresVal[i]))
        meanU = meanTrain[4]
        meanV = meanTrain[np.delete(nonNaNIndex, np.where(nonNaNIndex == 4))]  # delete all the index 5 from nonNaNIndex set
        #marginalize covariance matrix on above nonNaNIndex
        marginalCov = marg(covarianceMatrix,nonNaNIndex)
       # np.savetxt('marginalCov'+str(i)+ '.csv', marginalCov, delimiter=',')
        #convert covariannce to precision matrix
        precision = np.linalg.inv(marginalCov)
        #np.savetxt('precision'+str(i)+ '.csv', precision, delimiter=',')
        indexOfA = np.where(nonNaNIndex==4)
        A = precision[indexOfA,indexOfA]
        indexesOfB = np.array([ind for ind in range(len(precision[0])) if ind != indexOfA[0]])
        b = precision[indexOfA,indexesOfB];
        v =  featuresVal[i,np.delete(nonNaNIndex,np.where(nonNaNIndex==4))]#delete all the index 5 from nonNaNIndex set
        v = v - meanV
        bvMinusV = np.dot(b,v)
        prediction[i] = (meanU -  (np.linalg.inv(A) * bvMinusV))
        #print(prediction[i])
    featuresVal[0,4]  = meanU
    prediction[0]= meanU
    np.savetxt('prediction val.csv',prediction,delimiter=',')

    temp1 = [[el] for el in featuresVal[:,4]]
    temp2 = [[el] for el in prediction]
    print(((np.array(temp1) - np.array(temp2)) ** 2).mean(axis=0))
__main__()

