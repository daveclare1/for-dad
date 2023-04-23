from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

def dendrogram_from_df(df, method, cut_threshold=None):
    if cut_threshold is None:
        cut_threshold = 0.7     # default for dendrogram func anyway
    X = df.values
    Z = linkage(X, method=method)

    c, _ = cophenet(Z, pdist(X))

    fig = plt.figure(figsize=(10, 25))
    plt.title(f'Hierarchical Clustering, {method} method\
    \nCophenet: {c:.2f}\
    \nCut threshold: {cut_threshold}')
    plt.xlabel('Distance')
    plt.ylabel('Site')
    R = dendrogram(
        Z,
        orientation='right',
        leaf_font_size=8.,  # font size for the x axis labels,
        labels=df.index,
        truncate_mode=None,
        color_threshold=cut_threshold*max(Z[:,2]),
    )

    n_clusters = max([int(c[1]) for c in R['color_list']])
    
    return fig, Z, n_clusters