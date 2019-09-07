import random
import numpy
import matplotlib.pyplot as plt
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

points = numpy.array([[1, 2, 2], [3, 2, 2], [6, 5, 5], [6,5,1]])
MAX_ITER = 100
class KMeans:
    def __init__(self, inputDataFrame, numberOfClusters):
        self.inputData = inputDataFrame
        self.numberOfClusters = numberOfClusters
        self.numberOfInstances, self.numberOfFeatures = self.inputData.shape
        #print(self.numberOfInstances)
        #print(self.numberOfFeatures)
        #this saved states are fucked up we need to clean this to only the latest
        self.clusters_all_iterations_record = []

    def calculateEuclideanDistance(self,centroid, inputDataPoint):
        return numpy.linalg.norm(centroid - inputDataPoint)

    def generateRandomInitialCentroids(self):


        #we choose random centroids based on the number of clusters required
        randomCentroidsFromInputData = random.sample(list(self.inputData),self.numberOfClusters)
       # print(randomCentroidsFromInputData)
        return randomCentroidsFromInputData

    def assignDataPointToClosestCentroid(self,centroid):
        #clusterAllocationResult = defaultdict(list)
        clusterAllocationResult = defaultdict(list)
        for dataPoint in self.inputData:
            #below array is a temporary distance whose index will give the direct index of the centroid to which
            #a data point belongs to also this will reset to empty for next iteration
            euclideanDistanceBetweenDataPointAndCentroid= []

            for centroidPoint in centroid:
                #print("datapoint is")
                #print(dataPoint)
                #print("cluster point is")
                #print(centroidPoint)
                distance = self.calculateEuclideanDistance(centroid=centroidPoint, inputDataPoint=dataPoint)
                #print(distance)
                euclideanDistanceBetweenDataPointAndCentroid.append(distance)
            index_min = numpy.argmin(euclideanDistanceBetweenDataPointAndCentroid)
            #print("index minimum")
            #print(index_min)
            #dictionary cannot use a list as key but can use tuple as key
            #clusterAllocationResult[str(tuple(centroid[index_min]))].append(dataPoint)
            clusterAllocationResult[index_min].append(dataPoint)
        return clusterAllocationResult



    def runKMeansCoreAlgoritm(self):
        randomIntialCentroidsFromInputData = self.generateRandomInitialCentroids()
        print("random centroids are")
        print(randomIntialCentroidsFromInputData)
        #for iteration in range(self.MAX_ITER):
        iterationResults = self.assignDataPointToClosestCentroid(randomIntialCentroidsFromInputData)
        print(iterationResults)
        #proper membership print
        for x,y in iterationResults.items():
            print(randomIntialCentroidsFromInputData[x],y)



kmeansObj = KMeans(inputDataFrame=points, numberOfClusters=2)

kmeansObj.generateRandomInitialCentroids()
kmeansObj.runKMeansCoreAlgoritm()
