import random
import numpy
import matplotlib.pyplot as plt
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

points = numpy.array([[1, 2, 2], [3, 2, 2], [6, 5, 5], [6,5,1]])

class KMeans:
    def __init__(self, inputDataFrame, numberOfClusters):
        self.inputData = inputDataFrame
        self.numberOfClusters = numberOfClusters
        self.numberOfInstances, self.numberOfFeatures = self.inputData.shape
        self.centroid =[]
        self.MAX_ITER = 100
        #print(self.numberOfInstances)
        #print(self.numberOfFeatures)

    def calculateEuclideanDistance(self,centroid, inputDataPoint):
        return numpy.linalg.norm(centroid - inputDataPoint)

    #def recalculateCentroids(self,clusterAllocationResult):
    def isCentroidChanged(self,oldCentroid,newCentroid):
        if (self.calculateEuclideanDistance(oldCentroid,newCentroid) ==0):
            return False
        else:
            return True


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
        self.centroid = self.generateRandomInitialCentroids()
        print("random centroids are")
        print(self.centroid)
        for iteration in range(self.MAX_ITER):
            print("iterationcount")
            iterationResults = self.assignDataPointToClosestCentroid(self.centroid)
            print(iterationResults)
            newUpdatedCentroids = []
            #proper membership print
            for centroids,dataPoints in iterationResults.items():
                #print(randomIntialCentroidsFromInputData[x],y)
                newUpdatedCentroids.append(numpy.mean(dataPoints, axis=0))


            result = self.isCentroidChanged(self.centroid,numpy.array(newUpdatedCentroids))
            if result == True:

                print("algorithm not Converged")
                self.centroid =[]
                self.centroid=newUpdatedCentroids
                print("new updated centroids")
                print(newUpdatedCentroids)
            else:
                print("algorithm Converged")
                break;
        return iterationResults
        #updatedCentroids = self.recalculateCentroids(randomIntialCentroidsFromInputData)



    def plotKMeans(self,iterationResults, x_axis_label="", y_axis_label="", z_axis_label="", plot_title=""):

            fig = plt.figure()
            colors = ['r','g','b','y']
            ax = fig.add_subplot(111, projection='3d')
            print("new centroid")
            for centroidIndex,dataPoints in iterationResults.items():
                 print(self.centroid[centroidIndex],dataPoints)

            print("end of clustering results")
            for centroidIndex,dataPoints in iterationResults.items():
                print("final centroids")

                finalData = numpy.array(dataPoints)
                print("numpy datapoints")

                print(numpy.array(dataPoints))
                ax.scatter(finalData[:,0],finalData[:,1],finalData[:,2],c=colors[centroidIndex], marker='o' )
                ax.scatter(self.centroid[centroidIndex][0],self.centroid[centroidIndex][1],self.centroid[centroidIndex][2],c=colors[centroidIndex], marker='*');
               # ax.scatter(x, y, z, c='r', marker='o')

                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')

            plt.show()

kmeansObj = KMeans(inputDataFrame=points, numberOfClusters=2)

kmeansObj.generateRandomInitialCentroids()
iterationResults = kmeansObj.runKMeansCoreAlgoritm()
kmeansObj.plotKMeans(iterationResults)




