import os
import matplotlib as plt
import pickle
import seaborn as sns
import cv2
import numpy as np
from metrics import Metrics
import pandas as pd


# read the data
def read_pickle_file(filename):
    l = pickle.load(open(filename, 'rb'))
    g = l[0]
    path = l[1]
    return g, path

def save_pkl(i, arr):
    f = open("../outputs/all_data_"+str(i)+".pkl", 'wb')
    pickle.dump(arr, f)
    f.close()


# --- CODE RUNNING STARTS HERE --- # 

vw_sizes = {}
with open('../inputs/sizes.txt', 'r') as f:
    for line in f:
        l = line.strip().split()
        if len(l) == 2:
            vw_sizes[l[0]] = int(l[1])

all_data = []
i = 0

# Iterate through every single file in ../outputs. 
for placename in os.listdir("../outputs"):
    if placename[-4] == ".": # stupid way of checking if it's a file
        continue
    
    img = cv2.imread("../inputs/"+placename+".png")
    vw = vw_sizes[placename+".png"]


    for experimentType in os.listdir("../outputs/"+placename):
        filepath_experiment = "../outputs/"+placename+"/"+experimentType

        for nattempts in os.listdir(filepath_experiment):
            filepath_files = filepath_experiment + "/" + nattempts
            nats = nattempts
            percentage = 0
            if experimentType.__contains__("lawnmower"):
                percentage = nats
                nats = 0

            for f in os.listdir(filepath_files):
                if f[-4:] == ".png":
                    continue
                g, path = read_pickle_file(filepath_files+"/"+f)
                m = Metrics(img, vw)
                n, length, pc, plc, pwc = m.evaluate(g, path)

                row = [placename,  
                        experimentType, 
                        int(f[:-4]), 
                        int(nats), 
                        float(percentage), 
                        n, 
                        length, 
                        pc, 
                        plc, 
                        pwc]

                all_data.append(row)
                if len(all_data) > 300:
                    save_pkl(i, all_data)
                    i += 1
                    all_data = []

save_pkl(i, all_data)
exit
# fill dataframe
# Each row is one trial
# column titles "place", "experimentType", "trialnum", "nattempts", "percentScaling", nsamples, l, pc, lpc, w

df = pd.DataFrame(all_data, columns=["Place", "Experiment Type", "Trial Number", "nAttempts", \
    "Percent Scaling", "Graph Vertices", "Path Length", \
    "Percent Map Covered", "Percent Land Covered", "Percent Path Water"])

# rename our experiments
experiment_names = {
    "baseline": "Sampling Baseline", 
    "intelligent_neighbors": "Intelligent Neighbors", 
    "intelligent_distanced": "Intelligent Distanced", 
    "distributed_edges": "Brute Force Components", 
    "lawnmower_baseline": "Grid Baseline", 
    "lawnmower": "Lawnmower Intelligent"
}
df = df.replace(experiment_names)
# for oldName, newName in experiment_names.items():
#     df[df["Experiment Type"] == oldName]["Experiment Type"] = newName


f = open("../outputs/all_data.pkl", 'wb')
pickle.dump(df, f)
f.close()


# hue_order = ["Sampling Baseline", "Intelligent Neighbors", "Intelligent Distanced", "Distributed Edges", "Lawnmower Baseline", "Lawnmower Inteliigent"]

# boxplot_n150 = sns.boxplot(data=df[(df["nAttempts"]==10) or (df["Percentage"] == 1)], \
#                             x="Place", \
#                             y="Percent Map Covered", \
#                             hue="Experiment Type", \
#                             hue_order=hue_order)

# fig = boxplot_n150.get_figure()
# fig.savefig("out.png") 
