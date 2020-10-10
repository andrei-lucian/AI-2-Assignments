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
                print(self.clusters[x][y].prototype)

    def train(self):
        ## Step 1: initialize map with random vectors (A good place to do this, is in the initialisation of the clusters)
        ## Repeat 'epochs' times:
        ##     Step 2: Calculate the squareSize and the learningRate, these decrease linearly with the number of epochs.
        ##     Step 3: Every input vector is presented to the map (always in the same order)
        ##     For each vector its Best Matching Unit is found, and :
        ##         Step 4: All nodes within the neighbourhood of the BMU are changed, you don't have to use distance relative learning.
        ## Since training kohonen maps can take quite a while, presenting the user with a progress bar would be nice
        r = 0
        eta = 0 
        for currEpoch in range(self.epochs):
            eta = self.initial_learning_rate * (1 - currEpoch/self.epochs)
            r = self.n/2 * (1 - currEpoch/self.epochs)

            for i in range(self.n):
                for j in range(self.n):
                    self.clusters[i][j].current_members.clear()
            
            for i in range(len(self.traindata)-1):
                minX = 0
                minY = 0
                distance = 0
                minDistance = float('inf')
                for j in range(self.n):
                    for k in range(self.n):
                        for m in range(self.dim):
                            distance += math.pow(self.clusters[j][k].prototype[m] - self.traindata[i][m], 2)
                        distance = math.sqrt(distance)
                        if (distance <= minDistance):
                            minX = j
                            minY = k
                            minDistance = distance

                upperBoundX = min(self.n-1, minX + int(r))
                for n in range(max(0, minX - int(r)), upperBoundX):
                    upperBoundY = min(self.n - 1, minY + int(r))
                    for o in range(max(0, minY - int(r)), upperBoundY):
                        c = self.clusters[n][o]
                        newPrototype = []
                        for p in range(self.dim):
                            newPrototype.append(float((1-eta) * c.prototype[p] + eta * self.traindata[i][p]))
                        self.clusters[n][o].prototype = newPrototype
                self.clusters[minX][minY].current_members.add(i)

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