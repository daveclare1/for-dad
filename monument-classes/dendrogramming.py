from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

def dendrogram_from_df(df, method):
    X = df.values
    Z = linkage(X, method=method)

    c, _ = cophenet(Z, pdist(X))

    fig = plt.figure(figsize=(10, 25))
    plt.title(f'Hierarchical Clustering, {method} method\ncophenet: {c:.2f}')
    plt.xlabel('Distance')
    plt.ylabel('Site')
    dendrogram(
        Z,
        orientation='right',
        leaf_font_size=8.,  # font size for the x axis labels,
        labels=df.index,
        truncate_mode='level',
    )
    return fig, Z