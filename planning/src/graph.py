# https://medium.com/analytics-vidhya/graphs-in-python-adjacency-matrix-d0726620e8d7

import numpy as np
import utils
from scipy import sparse

class Graph:
    def __init__(self) -> None:
        self.vertices = []
        self.g = [[]]
        self.indices = {}
        self.edges = []
        self.numpy_verts = np.array([])

    def addVertex(self, v):
        if v in self.vertices:
            return

        nvertices_prev = len(self.vertices)
        self.vertices.append(v)

        if nvertices_prev == 0:
            self.g = np.array([[0]])
        else:
            g1 = np.zeros((len(self.vertices), len(self.vertices)))
            g1[:nvertices_prev, :nvertices_prev] = self.g
            self.g = g1.copy()

        self.indices[v] = len(self.vertices) - 1

        self.numpy_verts = np.empty(len(self.vertices), dtype=object)
        self.numpy_verts[:] = self.vertices

    def hasVertex(self, v):
        return v in self.vertices

    def addEdge(self, v1, v2, dist=None):
        # stupid thing
        if dist == None and type(v1) is tuple:
            dist = utils.euclidean(v1, v2)

        if not self.hasVertex(v1):
            # print(str(v1) + " not in graph.")
            return
        if not self.hasVertex(v2):
            # print(str(v2) + " not in graph.")
            return

        if self.hasEdge(v1, v2):
            return

        v1_index = self.indices[v1]
        v2_index = self.indices[v2]
        self.g[v1_index, v2_index] = dist
        self.g[v2_index, v1_index] = dist

        self.edges.append((v1, v2))
        
    def hasEdge(self, v1, v2):
        v1_index = self.indices[v1]
        v2_index = self.indices[v2]
        return self.g[v1_index, v2_index] > 0

    def getEdge(self, v1, v2):
        v1_index = self.indices[v1]
        v2_index = self.indices[v2]
        return self.g[v1_index, v2_index]

    def getAdjacencyMatrix(self):
        return self.g
    
    def getBooleanAdjacencyMatrix(self):
        return (self.g > 0).astype(np.int)

    def getNeighbors(self, v):
        index = self.indices[v]
        indices_of_neighbors = np.where(self.g[index, :] > 0)

        return self.numpy_verts[indices_of_neighbors]

    def getGraph(self):
        return self.g 
    
    def getEdges(self):
        return self.edges
                
    def getVertices(self):
        return self.vertices

    def nVertices(self):
        return len(self.vertices)

    def getLaplacianMatrix(self):
        # https://en.wikipedia.org/wiki/Laplacian_matrix 
        adj = self.getBooleanAdjacencyMatrix()
        degrees = np.diag(np.sum(adj, axis=0))
        
        return degrees - adj

    def getCCs(self):
        adj = self.getBooleanAdjacencyMatrix()
        n, labels = sparse.csgraph.connected_components(adj)
                
        components = []
        for i in range(n):
            comp = self.numpy_verts[np.where(labels == i)]
            components.append(list(comp))

        return components

    def euclidean(self, p1, p2):
        p1x, p1y = p1
        p2x, p2y = p2
        return np.sqrt((p1x-p2x)**2 + (p1y-p2y)**2)