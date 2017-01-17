# Dee Dong Explained me the process of how to approach the problem.
import numpy as np

import math
def marg(matrix, indexes):
    resCov = np.zeros([len(indexes), len(indexes)])
    for row, ind1 in enumerate(indexes):
        for col, ind2 in enumerate(indexes):
            resCov[row, col] = matrix[ind1, ind2]
    return resCov

def findNonNaN(f):
    nanIndex = np.isnan(f)
    index = []
    for i, val in enumerate(nanIndex):
        if (val == False):
            index.append(i)
    # print(index)
    return index

def __main__():
    # calculate mean_u and mean_v on the trainning set
    meanTrain = np.genfromtxt('mean.csv',delimiter=',')

    # Covariance matrix for train data
    covarianceMatrix = np.genfromtxt('covarianceMatrix.csv',delimiter=',')

    featuresVal = np.genfromtxt('test.csv', delimiter=',')
    featuresVal = featuresVal[:,:20]
    prediction = np.zeros(len(featuresVal))

    for i in range(len(featuresVal)):
        # value to be predicted
        featPredict = 4

        # find row indexes which we know
        nonNaNIndex = np.array(findNonNaN(featuresVal[i]))
        meanU = meanTrain[4]
        meanV = meanTrain[
            np.delete(nonNaNIndex, np.where(nonNaNIndex == 4))]  # delete all the index 5 from nonNaNIndex set
        # marginalize covariance matrix on above nonNaNIndex
        marginalCov = marg(covarianceMatrix, nonNaNIndex)
        # np.savetxt('marginalCov'+str(i)+ '.csv', marginalCov, delimiter=',')
        # convert covariannce to precision matrix
        precision = np.linalg.inv(marginalCov)
        # np.savetxt('precision'+str(i)+ '.csv', precision, delimiter=',')
        indexOfA = np.where(nonNaNIndex == 4)
        A = precision[indexOfA, indexOfA]
        indexesOfB = np.array([ind for ind in range(len(precision[0])) if ind != indexOfA[0]])
        b = precision[indexOfA, indexesOfB];
        v = featuresVal[
            i, np.delete(nonNaNIndex, np.where(nonNaNIndex == 4))]  # delete all the index 5 from nonNaNIndex set
        v = v - meanV
        bvMinusV = np.dot(b, v)
        prediction[i] = (meanU - (np.linalg.inv(A) * bvMinusV))
        # print(prediction[i])
    np.savetxt('prediction_test.csv', prediction, delimiter=',')

    #print(((np.array(temp1) - np.array(temp2)) ** 2).mean(axis=0))
__main__()

