from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tqdm import tqdm

def kmeans_clustering(df):
    embeddings = np.vstack(df.embedding.values)
    
    # Perform PCA
    pca = PCA(n_components=10)
    embeddings_pca = pca.fit_transform(embeddings)

    print(embeddings_pca.shape)
    # Fit the KMeans model to the PCA-reduced embeddings
    inertias = []
    ca = []
    cc = []
    kmeansobj = []
    for i in tqdm(range(50, 500)):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(embeddings_pca)
        inertias.append(kmeans.inertia_)
        print('Inertia', kmeans.inertia_)

        # Get the cluster assignments for each embedding
        cluster_assignments = kmeans.labels_

        # You can also get the cluster centers if you want
        cluster_centers = kmeans.cluster_centers_
        
        ca.append(cluster_assignments)
        cc.append(cluster_centers)
        kmeansobj.append(kmeans)
    
    return inertias, ca, cc, kmeansobj




embeddings = pd.read_pickle('embeddings.pkl')
inertias, ca, cc, kmeansobj = kmeans_clustering(embeddings)

plt.plot(range(50, 500), inertias)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.savefig('reduceddim_elbow_evaluation1_50.png')
plt.show()
