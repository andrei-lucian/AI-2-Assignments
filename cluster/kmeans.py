import random
import math
"""kmeans.py"""

class Cluster:
    """This class represents the clusters, it contains the
    prototype (the mean of all it's members) and memberlists with the
    ID's (which are Integer objects) of the datapoints that are member
    of that cluster. You also want to remember the previous members so
    you can check if the clusters are stable."""
    def __init__(self, dim):
        self.prototype = [0.0 for _ in range(dim)]
        self.current_members = set()
        self.previous_members = set()


class KMeans:
    def __init__(self, k, traindata, testdata, dim):
        self.traindata = traindata
        self.testdata = testdata
        self.dim = dim

        ## Threshold above which the corresponding html is prefetched
        self.prefetch_threshold = 0.5

        ## An initialized list of k clusters
        self.clusters = [Cluster(dim) for _ in range(k)]

        ## The accuracy and hitrate are the performance metrics (i.e. the results)
        self.accuracy = 0
        self.hitrate = 0

    def setCurrentMembers(self, cluster):
        ## Set current member depending on previous member
        cluster.previous_members.clear() 
        cluster.previous_members.update(cluster.current_members)
        cluster.current_members.clear()

    def calcPrototype(self):
        for cluster in (self.clusters): ## Loop through each cluster
            for member in (cluster.current_members): ## Loop through each member in each cluster
                for j in range(200): 
                    cluster.prototype[j] += self.traindata[member][j] ## Add all member vector values to prototype                
            for j in range(200):
                if (len(cluster.current_members) > 0):
                    cluster.prototype[j] /= len(cluster.current_members) ## Take the average for each prototype
            
            self.setCurrentMembers(cluster)

    def calcDistance(self):
        ## Loop through each data point in cluster
        for i in range(len(self.traindata)):
            minDistance = float('inf')
            minCluster = 0
            for k in range (len(self.clusters)):
                distance = 0
                ## Calculate distance for each data point
                for j in range(self.dim):
                    distance += math.pow(self.traindata[i][j] - self.clusters[k].prototype[j], 2)
                distance = math.sqrt(distance)
                ## Select cluster with smallest distance from data point
                if (distance < minDistance):
                    minCluster = k
                    minDistance = distance
            self.clusters[minCluster].current_members.add(i) ## Set cluster accordingly

    def train(self):
        ## Select an initial random partioning with k clusters
        for i in range(len(self.traindata)):
            self.clusters[random.randint(0, len(self.clusters)-1)].current_members.add(i)

        ## Calculate prototype and previousMembers for each cluster
        converged = False
        while not converged:
            self.calcPrototype()
            self.calcDistance()       
            ## Check if converged
            for cluster in self.clusters:
                if(cluster.current_members == cluster.previous_members):
                    converged = True
                else:
                    converged = False
        pass

    def test(self):
        hits = 0
        requests = 0
        prefetch = 0
        ## iterate along all clients. Assumption: the same clients are in the same order as in the testData
        for cluster in self.clusters:
            for member in cluster.current_members:
                for i in range (self.dim): ## iterate along all dimensions
                    if ((cluster.prototype[i] > self.prefetch_threshold) ## count number of hits
                        and (self.testdata[member][i] > self.prefetch_threshold)):
                        hits += 1 
                    if (self.testdata[member][i] > self.prefetch_threshold): ## count number of requests
                        requests += 1 
                    if (cluster.prototype[i] > self.prefetch_threshold): ## count prefetched htmls
                        prefetch += 1

        ## set the global variables hitrate and accuracy to their appropriate value
        self.accuracy = hits/prefetch
        self.hitrate = hits/requests
        pass

    def print_test(self):
        print("Prefetch threshold =", self.prefetch_threshold)
        print("Hitrate:", self.hitrate)
        print("Accuracy:", self.accuracy)
        print("Hitrate+Accuracy =", self.hitrate+self.accuracy)

    def print_members(self):
        for i, cluster in enumerate(self.clusters):
            print("Members cluster", i, ":", cluster.current_members)

    def print_prototypes(self):
        for i, cluster in enumerate(self.clusters):
            print("Prototype cluster", i, ":", cluster.prototype)