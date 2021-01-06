from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def table(df_X, kmeans):
    # Graph to visualize the table with clusters
    fig = plt.figure()
    ax = fig.add_subplot(111)

    scatter = ax.scatter(df_X['kmeans'], df_X.axes[0], s=1, c=kmeans[0])
    ax.set_title('K-Means Clustering')
    ax.set_xlabel('cluster')
    ax.set_ylabel('row')
    plt.colorbar(scatter, orientation="horizontal")
    plt.gca().invert_yaxis()

    # Display the output
    plt.show()