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
        ## Threshold above which the corresponding html is prefetched
        self.prefetch_threshold = 0.5
        self.initial_learning_rate = 0.8
        ## The accuracy and hitrate are the performance metrics (i.e. the results)
        self.accuracy = 0
        self.hitrate = 0
        ## Initialise N x N clusters
        for x in range(n):
            for y in range(n):
                self.clusters[x][y].prototype = self.traindata[r.randint(0,len(self.traindata)-1)]

    def train(self):
        ## Step 1: initialize map with random vectors (A good place to do this, is in the initialisation of the clusters)
        ## Repeat 'epochs' times:
        ##     Step 2: Calculate the squareSize and the learningRate, these decrease linearly with the number of epochs.
        ##     Step 3: Every input vector is presented to the map (always in the same order)
        ##     For each vector its Best Matching Unit is found, and :
        ##     Step 4: All nodes within the neighbourhood of the BMU are changed, you don't have to use distance relative learning.

        for curr_epoch in range(self.epochs):
            learning_rate = self.initial_learning_rate * (1 - curr_epoch/self.epochs)
            radius = self.n/2 * (1 - curr_epoch/self.epochs)

            for dim_1 in range(self.n):
                for dim_2 in range(self.n):
                    self.clusters[dim_1][dim_2].current_members.clear()
            
            for idx in range(len(self.traindata)-1):
                distance = 0
                min_dist = float('inf')
                for dim_1 in range(self.n):
                    for dim_2 in range(self.n):
                        for vect_dim in range(self.dim):
                            distance += math.pow(self.clusters[dim_1][dim_2].prototype[vect_dim] - self.traindata[idx][vect_dim], 2)
                        distance = math.sqrt(distance)
                        if (distance <= min_dist):
                            min_d1 = dim_1
                            min_d2 = dim_2
                            min_dist = distance

                upper_bound_d1 = min(self.n-1, min_d1 + int(radius))
                lower_bound_d1 = max(0, min_d1 - int(radius))
                upper_bound_d2 = min(self.n - 1, min_d2 + int(radius))
                lower_bound_d2 = max(0, min_d2 - int(radius))
                
                for dim_1 in range(lower_bound_d1, upper_bound_d1):
                    for dim_2 in range(lower_bound_d2, upper_bound_d2):
                        cluster = self.clusters[dim_1][dim_2]
                        new_prot = []
                        for vect_dim in range(self.dim):
                            new_prot.append(float((1-learning_rate) * cluster.prototype[vect_dim] + learning_rate * self.traindata[idx][vect_dim]))
                        self.clusters[dim_1][dim_2].prototype = new_prot
                self.clusters[min_d1][min_d2].current_members.add(idx)

    def test(self):
        ## iterate along all clients
        hits = 0
        requests = 0
        prefetch = 0

        ## for each client find the cluster of which it is a member
        for client in range(len(self.traindata)):
            for i in range(self.n):
                for j in range(self.n):
                    if (client in self.clusters[i][j].current_members):
                        owner = self.clusters[i][j]
            ## get the actual testData (the vector) of this client
            test = self.testdata[client]
            ## iterate along all dimensions
            for i in range(self.dim):
                if(owner.prototype[i] > self.prefetch_threshold and 
                    test[i] > self.prefetch_threshold):
                    hits += 1         ## count number of hits
                if(test[i] > self.prefetch_threshold):
                    requests += 1     ## count number of requests
                if(owner.prototype[i] > self.prefetch_threshold):
                    prefetch += 1     ## count prefetched htmls

        ## set the global variables hitrate and accuracy to their appropriate value
        self.accuracy = hits/prefetch
        self.hitrate = hits/requests

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