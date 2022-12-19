from planner import Planner
from metrics import Metrics
from graph import Graph
import numpy as np
import pickle
import sys, os
import time
    

def runExperiment(filename, visibilityWindow, experimentName, nExperiments=50, nsamples=50):
    lengths = []
    pcs = []
    lpcs = []
    water = []
    p = Planner(filename, visibilityWindow)
    m = Metrics(p.img, visibilityWindow)
    times = []
    foldername = "../outputs/"+filename[:-4]+"/"+experimentName+"/" +str(nsamples) + "/"
    if not os.path.exists(foldername):
        os.mkdir(foldername)
    for i in range(nExperiments):
        # if os.path.exists(foldername+str(i)+".pkl"):
        #     continue
        t = time.time()
        print("running exp")
        f = open(foldername+str(i)+".pkl", "rb")
        l = pickle.load(f)
        g = l[0]
        path = l[1]
        # g, path = p.plan(experimentName, nsamples)
   
        # f = open(foldername+str(i)+".pkl", "wb")
        # pickle.dump([g, path], f)
        # f.close()

        img = p.drawSamples(g.getVertices())
        # img = p.drawEdges(g.getEdges(), img)
        img = p.drawPath(path, img)
        p.writeImage(foldername+str(i)+"asdf", img)
        times.append(time.time()-t)
        n, l, pc, lpc, w = m.evaluate(g, path)
        print(n, l, pc, lpc, w)
        lengths.append(l)
        pcs.append(pc)
        lpcs.append(lpc)
        water.append(w)
    if len(times) != 0:
        avg = np.round(np.average(np.array(times)), 1)
        s = np.round(np.sum(np.array(times)), 1)
        print(avg, ", ", s)

    print(lengths, pcs, lpcs, water)
    return lengths, pcs, lpcs, water
    


if __name__ == "__main__":
    np.random.seed(1234)

    exps = ["baseline", "distributed_edges", "intelligent_distanced",\
            "intelligent_neighbors", "lawnmower", "lawnmower_baseline"]

    if not os.path.exists("../outputs"):
        os.mkdir("../outputs")

    filenames = ["place1.png", "place2.png", "place3.png"]
    
    lengths = {}
    with open('../inputs/sizes.txt', 'r') as f:
        for line in f:
            l = line.strip().split()
            if len(l) == 2:
                lengths[l[0]] = int(l[1])


    # plots we want:
    # percent coverage for 3 envs, comparison across all baslines
    # same for LPC

    # path length vs lpc plot.. manipulate nsamples 
    nsamples_list = [10, 15, 17, 20, 25, 50, 75, 100, 125, 150, 175, 200, 250]
    for filename in [filenames[0]]:
        print(filename)
        if not os.path.exists("../outputs/"+filename[:-4]):
            os.mkdir("../outputs/"+filename[:-4])

        for exp in exps:
            if not os.path.exists("../outputs/"+filename[:-4]+"/"+exp):
                os.mkdir("../outputs/"+filename[:-4]+"/"+exp)

        vw = lengths[filename]
        for nsamples in []:# nsamples_list:
            print(filename+" baseline", nsamples)
            runExperiment(filename, vw, "baseline", 15, nsamples=nsamples)
            print(filename+" intelligent_neighbors", nsamples)
            runExperiment(filename, vw, "intelligent_neighbors", 15, nsamples=nsamples)
            print(filename+" intelligent_distanced", nsamples)
            runExperiment(filename, vw, "intelligent_distanced", 15, nsamples=nsamples)
            if nsamples < 25:
                print(filename+" distributed_edges", nsamples)
                runExperiment(filename, vw, "distributed_edges", 15,nsamples=nsamples)
        for percentage in [0.35]:#[0.35, 0.45, 0.5, 0.75, 0.8, 1, 1.1, 1.25, 1.5, 1.75, 2, 2.5]:
            print(filename+" lawnmower_baseline", percentage)
            runExperiment(filename, vw, "lawnmower_baseline", 1, percentage)
            print(filename+" lawnmower", percentage)
            runExperiment(filename, vw, "lawnmower", 5, percentage)
