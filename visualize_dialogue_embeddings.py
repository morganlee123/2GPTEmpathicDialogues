#
# Generates 3-D UMAP Projections of the ChatGPT-Generated Dialogue Embeddings.
# Note: This script can also be used to do to same for the human-generated dialogue embedded representations.
# Code Author: Morgan Sandler (sandle20@msu.edu)
#

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from umap import UMAP  # Make sure to install umap-learn package
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


# support functions
def merge_categories(labels):
    category_map = {
    'afraid': 'negative',
    'angry': 'negative',
    'annoyed': 'negative',
    'anticipating': 'positive',
    'anxious': 'negative',
    'apprehensive': 'negative',
    'ashamed': 'negative',
    'caring': 'positive',
    'confident': 'positive',
    'content': 'positive',
    'devastated': 'negative',
    'disappointed': 'negative',
    'disgusted': 'negative',
    'embarrassed': 'negative',
    'excited': 'positive',
    'faithful': 'positive',
    'furious': 'negative',
    'grateful': 'positive',
    'guilty': 'negative',
    'hopeful': 'positive',
    'impressed': 'positive',
    'jealous': 'negative',
    'joyful': 'positive',
    'lonely': 'negative',
    'nostalgic': 'positive',
    'prepared': 'positive',
    'proud': 'positive',
    'sad': 'negative',
    'sentimental': 'positive',
    'surprised': 'positive',
    'terrified': 'negative',
    'trusting': 'positive'
}

    return [category_map.get(label, label) for label in labels]
def compute_dunn_index(embeddings, labels):
    unique_labels = np.unique(labels)
    if len(unique_labels) != 2:
        raise ValueError("There should be exactly two clusters.")

    # Separation: Minimum distance between clusters
    cluster_1 = embeddings[labels == unique_labels[0]]
    cluster_2 = embeddings[labels == unique_labels[1]]
    inter_cluster_distances = euclidean_distances(cluster_1, cluster_2)
    min_inter_cluster_distance = np.min(inter_cluster_distances)

    # Diameter: Maximum distance within a cluster
    intra_cluster_distance_1 = euclidean_distances(cluster_1, cluster_1)
    intra_cluster_distance_2 = euclidean_distances(cluster_2, cluster_2)
    max_intra_cluster_diameter = max(np.max(intra_cluster_distance_1), np.max(intra_cluster_distance_2))

    # Dunn Index
    dunn_index = min_inter_cluster_distance / max_intra_cluster_diameter
    return dunn_index

# Function to identify outlier indices
def find_outlier_indices(data, n_clusters=5, outlier_threshold=1.5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    cluster_centers = kmeans.cluster_centers_
    distances = cdist(data, cluster_centers, 'euclidean')
    min_distances = np.min(distances, axis=1)
    threshold = np.mean(min_distances) + outlier_threshold * np.std(min_distances)
    outlier_indices = np.where(min_distances > threshold)[0]
    return outlier_indices

# Function to visualize embeddings and identify outliers
def visualize_embeddings_umap(df):
    embeddings = np.vstack(df.embedding.values)
    umap = UMAP(n_components=3, random_state=42)
    embeddings_umap = umap.fit_transform(embeddings)

    original_labels = df.context.values
    merged_labels = merge_categories(original_labels)  # Ensure this function is defined
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(merged_labels)
    text_labels = label_encoder.inverse_transform(labels)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i, label in enumerate(text_labels):
        color = 'blue' if label == 'positive' else 'red'
        ax.scatter(embeddings_umap[i, 0], embeddings_umap[i, 1], embeddings_umap[i, 2], color=color, alpha=0.5)

    positive_patch = Patch(color='blue', label='Positive')
    negative_patch = Patch(color='red', label='Negative')
    ax.legend(handles=[positive_patch, negative_patch], title="Sentiment", loc='best')

    plt.savefig('FINAL_humangenerated_umap_viz_3D.pdf')
    plt.show()

    # Identify outlier indices
    outlier_indices = find_outlier_indices(embeddings_umap)
    outlier_original_labels = original_labels[outlier_indices]
    outlier_coordinates = embeddings_umap[outlier_indices]

    # Compute average coordinates for each unique label
    unique_labels = np.unique(outlier_original_labels)
    avg_coordinates = {label: np.mean(outlier_coordinates[outlier_original_labels == label], axis=0) for label in unique_labels}

    print("Average Coordinates for each Emotion in Outliers:")
    for label, coord in avg_coordinates.items():
        print(f"{label}: {coord}")

    return outlier_indices, outlier_original_labels, avg_coordinates

# main code
embeddings = pd.read_pickle('2GPTEmpathicDialoguesAsEmbeddings.pkl')
outlier_indices, outlier_labels = visualize_embeddings_umap(embeddings)

