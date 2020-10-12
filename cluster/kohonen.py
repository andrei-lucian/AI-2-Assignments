import random as r
import math
class Cluster:
    """This class represents the clusters, it contains the
    prototype (the mean of all it's members) and memberlists with the
    ID's (which are Integer objects) of the datapoints that are member
    of that cluster."""
    def __init__(self, dim):
        self.prototype = [0.0 for _ in range(dim)]
        self.current_members = set()

class Kohonen:
    def __init__(self, n, epochs, traindata, testdata, dim):
        self.n = n
        self.epochs = epochs
        self.traindata = traindata
        self.testdata = testdata
        self.dim = dim

        ## A 2-dimensional list of clusters. Size == N x N
        self.clusters = [[Cluster(dim) for _ in range(n)] for _ in range(n)]
        self.prefetch_threshold = 0.5    ## Threshold above which the corresponding html is prefetched
        self.initial_learning_rate = 0.8
       
        self.accuracy = 0                ## The accuracy and hitrate are the performance metrics (i.e. the results)
        self.hitrate = 0
        
        self.hits = 0                    ## Initialise hits, requests, and prefetches
        self.requests = 0
        self.prefetch = 0
        
        self.owner = Cluster(dim)        ## Initialise owner cluster object
        
        self.min_d1 = 0                  ## Initialise inidices of closest clusters
        self.min_d2 = 0
        
        self.upper_bound_d1 = 0          ## Initialise upper and lower bounds for neighbourhood of clusters
        self.lower_bound_d1 = 0
        self.upper_bound_d2 = 0
        self.lower_bound_d2 = 0
        
        self.radius = 0                  ## Initialise radius and learning rate
        self.learning_rate = 0
        
        self.currIdx = 0                 ## Initialise index of current training data vector
        
        for x in range(n):               ## Initialise map of N x N clusters with random input vectors
            for y in range(n):
                self.clusters[x][y].prototype = self.traindata[r.randint(0,len(self.traindata)-1)]

    ## Clear all clusters' current members
    def clearCurrentMembers(self):
        for dim_1 in range(self.n):
            for dim_2 in range(self.n):
                self.clusters[dim_1][dim_2].current_members.clear()

    ## Find an input vector's best mactching unit
    def findBMU(self):
        distance = 0
        min_dist = float('inf')
        for dim_1 in range(self.n):
            for dim_2 in range(self.n):
                for vect_dim in range(self.dim):
                    ## Calculate Euclidian distance between input vector and each cluster
                    distance += math.pow(self.clusters[dim_1][dim_2].prototype[vect_dim] 
                    - self.traindata[self.currIdx][vect_dim], 2)
                distance = math.sqrt(distance)
                ## Store the indices of the best matching unit (closest cluster)
                if (distance <= min_dist):
                    self.min_d1 = dim_1
                    self.min_d2 = dim_2
                    min_dist = distance

    ## Calculate the neighbourhood of a cluster
    def calculateNeighbourhood(self):
        self.upper_bound_d1 = min(self.n-1, self.min_d1 + int(self.radius)) 
        self.lower_bound_d1 = max(0, self.min_d1 - int(self.radius))
        self.upper_bound_d2 = min(self.n - 1, self.min_d2 + int(self.radius))
        self.lower_bound_d2 = max(0, self.min_d2 - int(self.radius))

    ## Update the prototype of a cluster
    def updateClusterPrototypes(self):
        for dim_1 in range(self.lower_bound_d1, self.upper_bound_d1):
            for dim_2 in range(self.lower_bound_d2, self.upper_bound_d2):
                cluster = self.clusters[dim_1][dim_2] 
                new_prot = [] 
                for vect_dim in range(self.dim): 
                    new_prot.append(float((1-self.learning_rate) * cluster.prototype[vect_dim] 
                    + self.learning_rate * self.traindata[self.currIdx][vect_dim]))
                self.clusters[dim_1][dim_2].prototype = new_prot

    ## Find the cluster that a client belongs to 
    def findOwner(self, client):
        for dim_1 in range(self.n):
            for dim_2 in range(self.n):
                if (client in self.clusters[dim_1][dim_2].current_members):
                    self.owner = self.clusters[dim_1][dim_2]

    ## Calculate the number of hits, requests, and prefetches
    def calculatePerformance(self, client):
        for vect_dim in range(self.dim):
            if(self.owner.prototype[vect_dim] > self.prefetch_threshold and 
                self.testdata[client][vect_dim] > self.prefetch_threshold):
                self.hits += 1         ## Count number of hits

            if(self.testdata[client][vect_dim] > self.prefetch_threshold):
                self.requests += 1     ## Count number of requests

            if(self.owner.prototype[vect_dim] > self.prefetch_threshold):
                self.prefetch += 1     ## Count prefetched URLs

    def train(self):
        ## Repeat 'epochs' times:
        for curr_epoch in range(self.epochs): 

            ## Calculate radius and learning_rate (both decrease linearly with epochs)
            self.learning_rate = self.initial_learning_rate * (1 - curr_epoch/self.epochs)
            self.radius = self.n/2 * (1 - curr_epoch/self.epochs)

            self.clearCurrentMembers()
            
            ## Each input vector is presented to the map
            for self.currIdx in range(len(self.traindata)-1): 

                ## For each vector its Best Matching Unit is found  
                self.findBMU()

                 ## Calculate neighbourhood of BMU                                 
                self.calculateNeighbourhood()    

                ## All clusters within the neighbourhood of the BMU are changed            
                self.updateClusterPrototypes()                

                ## Add input vector to its BMU
                self.clusters[self.min_d1][self.min_d2].current_members.add(self.currIdx)
            
    def test(self):
        for client in range(len(self.traindata)):
            self.findOwner(client) 
            self.calculatePerformance(client)

        ## Set the global variables hitrate and accuracy to their appropriate value
        self.accuracy = self.hits/self.prefetch
        self.hitrate = self.hits/self.requests

    def print_test(self):
        print("Prefetch threshold =", self.prefetch_threshold)
        print("Hitrate:", self.hitrate)
        print("Accuracy:", self.accuracy)
        print("Hitrate+Accuracy =", self.hitrate+self.accuracy)

    def print_members(self):
        for i in range(self.n):
            for j in range(self.n):
                print("Members cluster", (i, j), ":", self.clusters[i][j].current_members)

    def print_prototypes(self):
        for i in range(self.n):
            for j in range(self.n):
               print("Prototype cluster", (i, j), ":", self.clusters[i][j].prototype)