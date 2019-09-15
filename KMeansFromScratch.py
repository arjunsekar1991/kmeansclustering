import random
import numpy
import matplotlib.pyplot as plt
from array import *
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from scipy.spatial import distance

#points = numpy.array ([[1, 2, 2], [3, 2, 2], [6, 5, 5], [6, 5, 1]])
#all_NBME_avg_n4', 'all_PIs_avg_n131', 'HD_final'

class KMeans:
    def __init__(self, inputDataFrame, numberOfClusters):
        self.inputData = inputDataFrame
        self.numberOfClusters = numberOfClusters
        self.numberOfInstances, self.numberOfFeatures = self.inputData.shape
        self.centroid = []
        self.MAX_ITER = 100
        self.clusterMembership = []
        self.clusteredData = []
        # print(self.numberOfInstances)
        # print(self.numberOfFeatures)

    def calculateEuclideanDistance(self, centroid, inputDataPoint):
        return numpy.linalg.norm (centroid - inputDataPoint)

    # def recalculateCentroids(self,clusterAllocationResult):
    def isCentroidChanged(self, oldCentroid, newCentroid):
        if (self.calculateEuclideanDistance (oldCentroid, newCentroid) == 0):
            return False
        else:
            return True

    def generateRandomInitialCentroids(self):

        # we choose random centroids based on the number of clusters required
        randomCentroidsFromInputData = random.sample (list (self.inputData), self.numberOfClusters)
        # print(randomCentroidsFromInputData)
        return randomCentroidsFromInputData

    def assignDataPointToClosestCentroid(self, centroid):
        # clusterAllocationResult = defaultdict(list)
        clusterAllocationResult = defaultdict (list)
        for dataPoint in self.inputData:
            # below array is a temporary distance whose index will give the direct index of the centroid to which
            # a data point belongs to also this will reset to empty for next iteration
            euclideanDistanceBetweenDataPointAndCentroid = []

            for centroidPoint in centroid:
                # print("datapoint is")
                # print(dataPoint)
                # print("cluster point is")
                # print(centroidPoint)
                distance = self.calculateEuclideanDistance (centroid=centroidPoint, inputDataPoint=dataPoint)
                # print(distance)
                euclideanDistanceBetweenDataPointAndCentroid.append (distance)
            index_min = numpy.argmin (euclideanDistanceBetweenDataPointAndCentroid)
            # print("index minimum")
            # print(index_min)
            # dictionary cannot use a list as key but can use tuple as key
            # clusterAllocationResult[str(tuple(centroid[index_min]))].append(dataPoint)
            clusterAllocationResult[index_min].append (dataPoint)
        return clusterAllocationResult

    def runKMeansCoreAlgoritm(self):
        self.centroid = self.generateRandomInitialCentroids ()
        # print("random centroids are")
        # print(self.centroid)
        for iteration in range (self.MAX_ITER):
            # print("iterationcount")
            iterationResults = self.assignDataPointToClosestCentroid (self.centroid)
            # print(iterationResults)
            newUpdatedCentroids = []
            # proper membership print
            for centroids, dataPoints in iterationResults.items ():
                # print(randomIntialCentroidsFromInputData[x],y)
                newUpdatedCentroids.append (numpy.mean (dataPoints, axis=0))

            result = self.isCentroidChanged (self.centroid, numpy.array (newUpdatedCentroids))
            if result == True:

                # print("algorithm not Converged")
                self.centroid = []
                self.centroid = newUpdatedCentroids
            # print("new updated centroids")
            #  print(newUpdatedCentroids)
            else:
                # print("algorithm Converged")
                break;
        return iterationResults
        # updatedCentroids = self.recalculateCentroids(randomIntialCentroidsFromInputData)

    def calculateClusterMembership(self):
        for clusterid, dataPoints in iterationResults.items ():
            # print(clusterLabel,dataPoints)
            for temp in dataPoints:
                self.clusterMembership.append (clusterid)

    def plotKMeans(self, iterationResults, x_axis_label="", y_axis_label="", z_axis_label="", plot_title=""):

        fig = plt.figure ()
        colors = ['r', 'g', 'b', 'y']
        ax = fig.add_subplot (111, projection='3d')
        # print("new centroid")
        # for centroidIndex,dataPoints in iterationResults.items():
        # print(self.centroid[centroidIndex],dataPoints)

        # print("end of clustering results")
        for centroidIndex, dataPoints in iterationResults.items ():
            # print("final centroids")

            finalData = numpy.array (dataPoints)
            # print("numpy datapoints")

            # print(numpy.array(dataPoints))
            ax.scatter (finalData[:, 0], finalData[:, 1], finalData[:, 2], c=colors[centroidIndex], marker='o')
            ax.scatter (self.centroid[centroidIndex][0], self.centroid[centroidIndex][1],
                        self.centroid[centroidIndex][2], c=colors[centroidIndex], marker='*');
            # ax.scatter(x, y, z, c='r', marker='o')

            ax.set_xlabel ('X Label')
            ax.set_ylabel ('Y Label')
            ax.set_zlabel ('Z Label')

        plt.show ()

    def calculateClusterMembership(self):
        # clusterMembership = []
        for clusterid, dataPoints in iterationResults.items ():
            # print(clusterLabel,dataPoints)
            for temp in dataPoints:
                self.clusterMembership.append (clusterid)

    def computeSValue(self, index):
        sValue = 0.0
        for clusteredDataCount in self.clusteredData[index]:
            sValue += distance.euclidean (clusteredDataCount, self.centroid[index])

            norm_c = len (self.clusteredData[index])

        return sValue / norm_c

    def computeRIJValue(self, idx_i, idx_j):
        RIJ = 0
        try:
            interClusterDistance = self.calculateEuclideanDistance (self.centroid[idx_i], self.centroid[idx_j])
            RIJ = (self.computeSValue (idx_i) + self.computeSValue (idx_j)) / interClusterDistance
            # print(RIJ)
        except:
            return 0
        return RIJ

    def computeRValue(self, clusterCounter):
        Rij = []
        clusterIndexCombination = numpy.arange (self.numberOfClusters)
        idx_i = clusterCounter

        for idx_j, j in enumerate (clusterIndexCombination):
            if (idx_i != idx_j):
                # Rij.append((i, j))
                #  print("i ", idx_i, "j  ",idx_j)
                Rij.append (self.computeRIJValue (idx_i, idx_j))
        return max (Rij)

    def daviesBouldinIndex(self):
        print ("calculation started")
        rValue = 0.0
        for clusterCounter in range (self.numberOfClusters):
            rValue = rValue + self.computeRValue (clusterCounter)
            # print(rValue)
        rValue = float (rValue) / float (self.numberOfClusters)
        return rValue



#kmeansObj.plotKMeans(iterationResults)

import csv

output = []

#f = open( 'BSOM_DataSet_revised.csv', 'r' ) #open the file in read universal mode
import pandas as pd
fields = ['all_NBME_avg_n4', 'all_PIs_avg_n131', 'HD_final']

df = pd.read_csv('BSOM_DataSet_revised.csv', skipinitialspace=True, usecols=fields)
# See the keys
print (df.keys())
# See content in 'star_name'
print(len(df))
#for dataframeCounter in range(len (df)):
#    print(df.all_NBME_avg_n4[dataframeCounter],"----",df.all_PIs_avg_n131[dataframeCounter],'-------',df.HD_final[dataframeCounter])
#df.to_numpy()
print(df.to_numpy())
points =df.to_numpy()

kmeansObj = KMeans (inputDataFrame=points, numberOfClusters=6)

kmeansObj.generateRandomInitialCentroids ()
iterationResults = kmeansObj.runKMeansCoreAlgoritm ()
clusterLabels = kmeansObj.calculateClusterMembership ()
print ("iteation results", iterationResults)
kmeansObj.clusteredData = [iterationResults[k] for k in range(kmeansObj.numberOfClusters)]
print("clustered Data",kmeansObj.clusteredData)
print (kmeansObj.daviesBouldinIndex())

