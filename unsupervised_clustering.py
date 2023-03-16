# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:21:03 2023

@author: Sebastian
"""

import glob

# Plotting
import seaborn as sb
from matplotlib import pyplot as plt
import cv2

# Machine learning 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn import cluster, mixture
import pandas as pd

# Notification while models run
from colorama import Fore, Style
import datetime as time
import warnings

DATA_DIRECTORY = "data/"
OUTPUT_DIRECTORY = "output/"
# %% Import data
# For suppressing warnings

# For keeping track of which models have run

# Machine learning
#from sklearn.preprocessing import OrdinalEncoder

# Plotting

# %% Read in data
food_data = pd.read_csv(DATA_DIRECTORY + "food_data.csv")

# Logic for filtering nutrients
#food_col_regex = re.compile("^[^-].+?_100g$")
#[col for col in list(food_data.columns) if food_col_regex.match(col)]
#na_counts = food_data.isna().sum()
#na_counts.filter(regex = food_col_regex).sort_values().to_csv("test.csv")

# Picking these nutrients because afterwards there's a real jump in missing values
with(open(DATA_DIRECTORY + "nutrients.txt", "r")) as nutrient_file:
    nutrients = nutrient_file.read().splitlines()

XY = food_data[nutrients + ["nutrition_grade_fr"]].dropna()

X = XY[nutrients]
Y = XY["nutrition_grade_fr"]

# %% Perform clustering

# Does the data naturally cluster into nutriscores?

# Create parameters for clustering

params = {
    "quantile": 0.3,
    "eps": 0.3,
    "damping": 0.9,
    "preference": -200,
    "n_neighbors": 3,
    "n_clusters": 5,
    "min_samples": 7,
    "xi": 0.05,
    "min_cluster_size": 0.1,
}

# normalize dataset for easier parameter selection
X = StandardScaler().fit_transform(X)

# estimate bandwidth for mean shift
bandwidth = cluster.estimate_bandwidth(X, quantile=params["quantile"])

# connectivity matrix for structured Ward
connectivity = kneighbors_graph(
    X, n_neighbors=params["n_neighbors"], include_self=False
)
# make connectivity symmetric
connectivity = 0.5 * (connectivity + connectivity.T)


# Create cluster objects

ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
two_means = cluster.MiniBatchKMeans(
    n_clusters=params["n_clusters"], n_init="auto")
ward = cluster.AgglomerativeClustering(
    n_clusters=params["n_clusters"], linkage="ward", connectivity=connectivity
)
spectral = cluster.SpectralClustering(
    n_clusters=params["n_clusters"],
    eigen_solver="arpack",
    affinity="nearest_neighbors",
)
dbscan = cluster.DBSCAN(eps=params["eps"])
optics = cluster.OPTICS(
    min_samples=params["min_samples"],
    xi=params["xi"],
    min_cluster_size=params["min_cluster_size"],
)
affinity_propagation = cluster.AffinityPropagation(
    damping=params["damping"], preference=params["preference"], random_state=0
)
average_linkage = cluster.AgglomerativeClustering(
    linkage="average",
    metric="manhattan",
    n_clusters=params["n_clusters"],
    connectivity=connectivity,
)
birch = cluster.Birch(n_clusters=params["n_clusters"])
gmm = mixture.GaussianMixture(
    n_components=params["n_clusters"], covariance_type="full"
)

clustering_algorithms = (
    ("MiniBatch KMeans", two_means),
    ("Affinity Propagation", affinity_propagation),
    ("MeanShift", ms),
    ("Spectral Clustering", spectral),
    ("Ward", ward),
    ("Agglomerative Clustering", average_linkage),
    ("DBSCAN", dbscan),
    ("OPTICS", optics),
    ("BIRCH", birch),
    ("Gaussian Mixture", gmm),
)

algo_dict = {}
for name, algorithm in clustering_algorithms:
    #t0 = time.time()
    print("\n[", time.datetime.now(), "]",
          f"{Fore.RED}****{name}****{Style.RESET_ALL}")
    # catch warnings related to kneighbors_graph
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="the number of connected components of the "
            + "connectivity matrix is [0-9]{1,2}"
            + " > 1. Completing it to avoid stopping the tree early.",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Graph is not fully connected, spectral embedding"
            + " may not work as expected.",
            category=UserWarning,
        )
        algorithm.fit(X)

    #t1 = time.time()
    if hasattr(algorithm, "labels_"):
        y_pred = algorithm.labels_.astype(int)
    else:
        y_pred = algorithm.predict(X)

    algo_dict[name] = y_pred

# %% Examine clusters
nutriscore_clusters = pd.DataFrame(
    algo_dict | {"nutriscore": food_data.loc[Y.index, "nutrition_grade_fr"].array})

mb_crosstab = pd.crosstab(
    nutriscore_clusters["MiniBatch KMeans"], nutriscore_clusters["nutriscore"])

# sb.heatmap(mb_crosstab, cmap = "Reds")
# sb.clustermap(mb_crosstab, col_cluster = False, cmap = "Reds")
# sb.clustermap(mb_crosstab, col_cluster = False, z_score = 1, cmap = "Reds")

for algo in algo_dict.keys():
    mb_crosstab = pd.crosstab(
        nutriscore_clusters[algo], nutriscore_clusters["nutriscore"])
    sb.clustermap(mb_crosstab, col_cluster=False, z_score=1, cmap="Reds")
    plt.title(algo)
    
    plt.savefig(DATA_DIRECTORY + "temp_" + algo + ".png")

im_h1 = list()
im_h2 = list()

for index,image in enumerate(glob.glob(DATA_DIRECTORY + "temp_*")):
    if index < 5:
        im_h1.append(cv2.imread(image))
    else:
        im_h2.append(cv2.imread(image))
        
img_h1 = cv2.hconcat(im_h1)
img_h2 = cv2.hconcat(im_h2)

img = cv2.vconcat([img_h1, img_h2])

cv2.imwrite(OUTPUT_DIRECTORY + "cluster_analysis.png", img)

#' So clearly, without training any models on the output data, we can see that nutriscores - at least the gap between ABC and DE form a natural grouping.
