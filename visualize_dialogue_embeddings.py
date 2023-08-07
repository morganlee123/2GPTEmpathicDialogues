from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def visualize_embeddings(df):
    embeddings = np.vstack(df.embedding.values)

    # Perform PCA
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)

    print(embeddings_pca.shape)

    # Fit the KMeans model to the PCA-reduced embeddings
    kmeans = KMeans(n_clusters=32)
    kmeans.fit(embeddings_pca)

    # Get the cluster assignments for each embedding
    emotion_categories = kmeans.labels_

    plt.figure(figsize=(10, 8))

    # Use emotion_categories as color labels in the scatter plot
    scatter = plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c=emotion_categories, cmap='rainbow')
    
    # Create a legend for the colors
    legend1 = plt.legend(*scatter.legend_elements(), title="Emotion Categories")
    plt.gca().add_artist(legend1)

    plt.savefig('gpt_viz.png')
    plt.show()

    # Compare with original labels
    original_labels = df.context.values  # Assuming 'context' column has the original labels

    # Convert original labels to numbers
    le = LabelEncoder()
    original_labels_encoded = le.fit_transform(original_labels)

    cm = confusion_matrix(original_labels_encoded, emotion_categories)
    
    # Visualize the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('gpt_conf_mat.png')
    plt.show()

    # Get the emotion labels that most often correspond to each KMeans cluster
    most_common_emotions = np.argsort(cm, axis=0)[-3:][::-1]

    # Compute the percentages
    percentages = np.sort(cm / cm.sum(axis=1, keepdims=True), axis=0)[-3:][::-1] * 100

    # Print out the results
    for i in range(cm.shape[1]):
        top_emotions = le.inverse_transform(most_common_emotions[:, i])
        top_percentages = percentages[:, i]
        print(f"CLUSTER CATEGORY {i + 1} - {', '.join([f'{emotion} ({percentage:.2f}%)' for emotion, percentage in zip(top_emotions, top_percentages)])}")




embeddings = pd.read_pickle('gpt_embeddings.pkl')

#embeddings['newembed'] = embeddings['embedding'].apply(string_to_numpy_array)

visualize_embeddings(embeddings)


