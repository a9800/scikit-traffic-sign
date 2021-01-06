import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from table import table
from pixels import pix
from sklearn.metrics.cluster import homogeneity_score


def runKMeans(value,clusters):
    model = KMeans(n_clusters=clusters, random_state=0).fit(value)
    return (model.predict(value), model.cluster_centers_)


print("Reading files and running k means..")

# Read in the csv
df_X = pd.read_csv("../../data/x_train_gr_smpl.csv")
df_y = pd.read_csv("../../data/y_train_smpl.csv")

# Run K-means
labels, center = runKMeans(df_X,10)

# Add the labels back to dataframe
kmeans = pd.DataFrame(labels)
df_X.insert((df_X.shape[1]),'kmeans',kmeans)

# Calculate the accuracy
print("Accuracy: ",  homogeneity_score(np.concatenate(kmeans.to_numpy()),np.concatenate(df_y.to_numpy())))

# Create the two graphs
print("Creating table of clusters...")
table(df_X,kmeans)

print("Visualizing clusters...")
pix(df_X)