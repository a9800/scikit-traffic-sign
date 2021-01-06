from sklearn.decomposition import PCA
import pylab as pl


def pix(df_X):

    # Create the scatter plot
    pca = PCA(n_components=2).fit(df_X)
    pca_2d = pca.transform(df_X)

    colorTable = [
        "#000F08",
        "#FD3E81", 
        "#12355B", 
        "#1B998B", 
        "#420039", 
        "#FFB17A", 
        "#3E885B", 
        "#22181C", 
        "#FCE762", 
        "#6E44FF"
    ]

    # Set the colours for each point
    for i in range(0, pca_2d.shape[0]): 
        pl.scatter(pca_2d[i,0],pca_2d[i,1],c=colorTable[df_X.loc[i,"kmeans"]], marker='x')

    # Display the output 
    pl.show()