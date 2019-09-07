import random
import numpy
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

points = numpy.array([[1, 2, 2], [3, 2, 2], [6, 5, 5], [6,5,1]])
MAX_ITER = 100
class KMeans:
    def __init__(self, inputDataFrame, numberOfClusters):
        self.inputData = inputDataFrame
        self.numberOfClusters = numberOfClusters
        self.numberOfInstances, self.numberOfFeatures = self.inputData.shape
        print(self.numberOfInstances)
        print(self.numberOfFeatures)
        #this saved states are fucked up we need to clean this to only the latest
        self.clusters_all_iterations_record = []

    def get_euclidean_distance(centroid, inputDataPoint):
        return numpy.linalg.norm(centroid - inputDataPoint)

    def generateRandomInitialCentroids(self):

        #random_dataset_indices = random.sample(range(0, self.number_of_instances), self.k_number_of_clusters)
        #we choose random centroids based on the number of clusters required
        randomCentroidsFromInputData = random.sample(list(self.inputData),self.numberOfClusters)
        print(randomCentroidsFromInputData)
        #return new_array



kmeansObj = KMeans(inputDataFrame=points, numberOfClusters=2)
kmeansObj.generateRandomInitialCentroids()
