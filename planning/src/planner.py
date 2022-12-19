from sampler import Samplers
from isvalids import Valids
from graph import Graph
from edges import GraphEdgeMaker
import cv2

RED_HSV = (0, 100, 100)
RED_RGB = (255, 0, 0)
RED_BGR = (0, 0, 255)
BLACK_BGR = (0, 0, 0)

class Planner:
    def __init__(self, filename, visibilityWindow):
        self.img = cv2.imread("../inputs/"+filename, cv2.IMREAD_COLOR)
        self.visibilityWindow = visibilityWindow

    def plan(self,mode, nsamples=50):
        if mode == "baseline":
            sampler = Samplers(self.img).uniform
            vc = Valids(self.img, self.visibilityWindow).alwaysTrue
            edgeMaker = GraphEdgeMaker(self.img, self.visibilityWindow, vc).nearestNeighborTSP
            # return self.baseline(nsamples)

        elif mode == "intelligent_neighbors":
            sampler = Samplers(self.img).uniform
            vc = Valids(self.img, self.visibilityWindow).notOnWater
            edgeMaker = GraphEdgeMaker(self.img, self.visibilityWindow, vc).smartTSP

        elif mode == "intelligent_distanced":
            sampler = Samplers(self.img).uniform
            vc = Valids(self.img, self.visibilityWindow).notOnWater_distanced
            edgeMaker = GraphEdgeMaker(self.img, self.visibilityWindow, vc).smartTSP

        elif mode == "distributed_edges":
            sampler = Samplers(self.img).uniform
            vc = Valids(self.img, self.visibilityWindow).notOnWater_distanced
            edgeMaker = GraphEdgeMaker(self.img, self.visibilityWindow, vc).localNetworkAlgorithm
            # return self.dumbSamplerSmartEdges(nsamples)

        elif mode == "lawnmower":
            sampler = Samplers(self.img, self.visibilityWindow).grid
            vc = Valids(self.img, self.visibilityWindow).notOnWater
            edgeMaker = GraphEdgeMaker(self.img, self.visibilityWindow, vc).lawnmower

        elif mode == "lawnmower_baseline":
            sampler = Samplers(self.img, self.visibilityWindow).grid
            vc = Valids(self.img, self.visibilityWindow).alwaysTrue
            edgeMaker = GraphEdgeMaker(self.img, self.visibilityWindow, vc).lawnmower

        return self.planner(sampler, vc, edgeMaker, nsamples)


    ### MAKE SOME PLANZ ### 
    def planner(self, sampler, vc, edgeMaker, nsamples=50):
        g = Graph()
        samples = sampler(vc, nsamples)
        for sample in samples:
            g.addVertex(sample)

        g, path, length = edgeMaker(samples, g)
        return g, path

    def baseline(self, nsamples=50):
        g = Graph()
        sampler = Samplers(self.img)
        valid = Valids(self.img)

        # randomly sample points
        samples = sampler.uniform(valid.notOnWater, nsamples)
        for sample in samples:
            g.addVertex(sample)

        # Greedy TSP path
        e = GraphEdgeMaker(self.img)
        g, path, length = e.nearestNeighborTSP(samples, g, useGraphEdges=False)

        return g, path

    def dumbSamplerSmartEdges(self, nsamples=50):
        g = Graph()

        # Randomly sample points
        sampler = Samplers(self.img)
        valid = Valids(self.img)

        samples = sampler.uniform(valid.notOnWater, nsamples)
        for sample in samples:
            g.addVertex(sample)

        # write the image of just the samples
        samples_img = self.drawSamples(samples)
        self.writeImage("mine_samples", samples_img)

        # Find path. 
        e = GraphEdgeMaker(self.img)
        g, path, length = e.localNetworkAlgorithm(samples, g)

        # Save path as img (for future reference)
        graphimg = self.drawSamples(g.getVertices())
        graphimg = self.drawEdges(g.getEdges(), graphimg)
        graphimg = self.drawPath(path, graphimg)
        self.writeImage("mine_edges", graphimg)

        return g, path

    #### ARTSY STUFF ####
    def drawSamples(self, samples, img=None):
        if img is None:
            img = self.img.copy()
        for pt in samples:
            ptx, pty = pt  
            img = cv2.circle(img, (pty, ptx), color=RED_BGR, radius=5, thickness=-1)
        return img

    def drawEdge(self, img, v1, v2, color, thickness=2):
        v1x, v1y = v1
        v2x, v2y = v2
        return cv2.line(img, (v1y, v1x), (v2y, v2x), color=color, thickness=thickness)
        
    def drawEdges(self, edges, img=None):
        if img is None:
            img = self.img.copy()
        for edge in edges:
            img = self.drawEdge(img, edge[0], edge[1], RED_BGR)
            
        return img

    def drawPath(self, path, img=None):
        if img is None:
            img = self.img.copy()

        for i in range(len(path)-1):
            v1 = path[i]
            v2 = path[i+1]
            img = self.drawEdge(img, v1, v2, BLACK_BGR, thickness=5)
        return img

    def drawPathStart(self, pathStart, img=None):
        if img is None:
            img = self.img.copy()
        ptx, pty = pathStart
        img = cv2.circle(img, (pty, ptx), radius=10, color=BLACK_BGR, thickness=-1)
        return img

    def writeImage(self, filename, img):
        full_filename = "../outputs/"+filename+".png"
        cv2.imwrite(full_filename, img)
