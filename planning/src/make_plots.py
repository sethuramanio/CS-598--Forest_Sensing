import seaborn as sns
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np


cols = ["Place", "Experiment Type", "Trial Number", "nAttempts", \
    "Percent Scaling", "Graph Vertices", "Path Length", \
    "Percent Map Covered", "Percent Land Covered", "Percent Path Land"]
df = pd.DataFrame(columns=cols)

for i  in range(7):
    rows = pickle.load(open("../outputs/all_data_"+str(i)+'.pkl', 'rb'))
    temp = pd.DataFrame(rows, columns=cols)
    # print(temp)
    df = pd.concat([df, temp])

# print(df.shape)

# rename our experiments
experiment_names = {
    "baseline": "Sampling Baseline", 
    "intelligent_neighbors": "Intelligent Neighbors", 
    "intelligent_distanced": "Intelligent Distanced", 
    "distributed_edges": "Brute Force Components", 
    "lawnmower_baseline": "Grid Baseline", 
    "lawnmower": "Grid Intelligent"
}
df = df.replace(experiment_names)

# print(df[df["Experiment Type"] == "Grid Baseline"])

hue_order = ["Sampling Baseline", "Intelligent Neighbors", "Intelligent Distanced", "Grid Baseline", "Grid Intelligent", "Brute Force Components"]

p1_good = 150
p2_good = 175
p3_good = 250

df150p1 = df.loc[(df["Place"] == "place1") & ((df["nAttempts"]==p1_good) | ((df["Graph Vertices"] < 1.2*p1_good) & (df["Graph Vertices"] > 0.8* p1_good)))]
df150p2 = df.loc[(df["Place"] == "place2") & ((df["nAttempts"]==p2_good) | ((df["Graph Vertices"] < 1.2*p2_good) & (df["Graph Vertices"] > 0.8* p2_good)))]
df150p3 = df.loc[(df["Place"] == "place3") & ((df["nAttempts"]==p3_good) | ((df["Graph Vertices"] < 1.2*p3_good) & (df["Graph Vertices"] > 0.8* p3_good)))]
df150 = pd.concat([df150p1, df150p2, df150p3])

# print(df150)

plt.figure()
ax=plt.subplot(111)
plt.figure()
boxplot_n150_path = sns.barplot(data=df150, \
                            x="Place", \
                            y="Percent Land Covered", \
                            hue="Experiment Type", \
                            hue_order=hue_order)
plt.title("Percent of the environment's land that is seen on the path. \n(Higher is better)")
plt.savefig("barplot_plc.png") 

plt.figure()
boxplot_n150_path = sns.barplot(data=df150, \
                            x="Place", \
                            y="Percent Path Land", \
                            hue="Experiment Type", \
                            hue_order=hue_order)
plt.title("Percent of path over land, for a reasonable number of vertices. \n(Higher is better)")
plt.savefig("barplot_ppl.png") 

plt.figure()
boxplot_n150_path = sns.barplot(data=df150, \
                            x="Place", \
                            y="Path Length", \
                            hue="Experiment Type", \
                            hue_order=hue_order)
plt.title("Path length for a reasonable number of vertices.")
plt.savefig("barplot_length.png") 

plt.figure()
place1_data = df[df["Place"] == "place1"]
print(np.max(place1_data["Path Length"]))
scatter = sns.scatterplot(data=place1_data, \
        x="Path Length", \
            y="Percent Land Covered",\
            hue="Experiment Type",  hue_order=hue_order, \
            s=20)
plt.title("Path Length vs Land Covered for each strategy, location 1")
plt.savefig("scatter.png") 
