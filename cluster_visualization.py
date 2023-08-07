# Author: Morgan Sandler (sandle20@msu.edu)
# Purpose: Viz learned (unsupervised) clusters of emotion categories

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# GPT DIALOGUES:

all_clusters_data = [
    ('angry', 15.92), ('sad', 14.45), ('devastated', 14.14),
    ('surprised', 12.27), ('grateful', 10.23), ('proud', 10.03),
    ('surprised', 18.82), ('sentimental', 15.13), ('nostalgic', 13.41),
    ('jealous', 12.11), ('lonely', 11.94), ('apprehensive', 10.11),
    ('confident', 16.55), ('hopeful', 15.60), ('prepared', 10.59),
    ('anxious', 16.43), ('lonely', 9.03), ('apprehensive', 7.17),
    ('terrified', 15.40), ('disgusted', 12.30), ('afraid', 10.58),
    ('lonely', 13.99), ('jealous', 12.97), ('apprehensive', 11.88),
    ('guilty', 18.50), ('devastated', 17.71), ('ashamed', 15.53),
    ('proud', 27.43), ('excited', 20.99), ('joyful', 19.03),
    ('excited', 34.26), ('anticipating', 23.53), ('joyful', 21.54),
    ('disappointed', 8.00), ('grateful', 6.55), ('terrified', 6.47),
    ('nostalgic', 58.49), ('sentimental', 25.16), ('impressed', 9.57),
    ('furious', 20.43), ('angry', 15.70), ('embarrassed', 12.43),
    ('grateful', 12.61), ('surprised', 9.70), ('jealous', 9.64),
    ('sad', 14.56), ('devastated', 13.96), ('annoyed', 11.22),
    ('prepared', 11.90), ('confident', 9.86), ('hopeful', 8.16),
    ('annoyed', 11.38), ('disappointed', 10.36), ('afraid', 8.87),
    ('confident', 18.31), ('proud', 15.83), ('joyful', 10.41),
    ('surprised', 13.95), ('impressed', 13.30), ('excited', 10.42),
    ('lonely', 10.91), ('disappointed', 10.75), ('anxious', 10.31),
    ('guilty', 20.77), ('furious', 18.81), ('angry', 15.54),
    ('surprised', 10.15), ('sentimental', 8.86), ('nostalgic', 5.19),
    ('joyful', 15.44), ('impressed', 15.25), ('surprised', 7.72),
    ('confident', 13.73), ('grateful', 11.41), ('proud', 11.15),
    ('disgusted', 21.09), ('embarrassed', 20.19), ('angry', 9.76),
    ('grateful', 11.43), ('trusting', 10.22), ('surprised', 9.85),
    ('anxious', 16.61), ('lonely', 15.02), ('apprehensive', 11.16),
    ('disgusted', 31.28), ('terrified', 6.75), ('afraid', 2.73),
    ('disappointed', 16.00), ('lonely', 11.84), ('devastated', 11.43),
    ('annoyed', 13.74), ('furious', 12.68), ('terrified', 12.11),
    ('hopeful', 9.40), ('anxious', 8.74), ('confident', 7.99),
]

"""


# HUMAN DIALOGUES:

all_clusters_data = [
    ('embarrassed', 20.19), ('disgusted', 16.70), ('furious', 11.93),
    ('lonely', 11.26), ('jealous', 9.51), ('surprised', 9.01),
    ('jealous', 8.31), ('anxious', 7.65), ('confident', 6.87),
    ('excited', 13.46), ('joyful', 11.81), ('surprised', 8.38),
    ('disgusted', 10.72), ('terrified', 10.38), ('angry', 7.38),
    ('sad', 8.54), ('devastated', 6.99), ('afraid', 5.80),
    ('confident', 17.78), ('hopeful', 9.93), ('prepared', 9.11),
    ('nostalgic', 15.86), ('sentimental', 15.68), ('caring', 6.92),
    ('sad', 10.36), ('lonely', 10.02), ('disappointed', 9.90),
    ('excited', 17.06), ('anticipating', 16.58), ('proud', 12.03),
    ('annoyed', 11.22), ('angry', 9.08), ('guilty', 8.74),
    ('anxious', 9.50), ('disappointed', 8.74), ('apprehensive', 8.00),
    ('afraid', 12.97), ('disgusted', 12.30), ('terrified', 12.11),
    ('excited', 15.60), ('proud', 14.42), ('anticipating', 14.08),
    ('nostalgic', 10.57), ('lonely', 10.48), ('sentimental', 9.78),
    ('disappointed', 6.91), ('annoyed', 5.61), ('embarrassed', 5.24),
    ('confident', 12.50), ('hopeful', 9.40), ('prepared', 7.99),
    ('surprised', 9.54), ('jealous', 9.33), ('lonely', 7.85),
    ('sad', 7.88), ('afraid', 6.83), ('terrified', 6.23),
    ('excited', 9.63), ('proud', 9.40), ('anticipating', 9.04),
    ('sad', 10.29), ('devastated', 9.85), ('annoyed', 7.34),
    ('surprised', 13.47), ('nostalgic', 11.35), ('impressed', 9.09),
    ('proud', 8.44), ('joyful', 7.68), ('excited', 6.85),
    ('terrified', 13.01), ('annoyed', 12.78), ('guilty', 12.63),
    ('disappointed', 12.00), ('anxious', 8.74), ('apprehensive', 8.62),
    ('embarrassed', 17.48), ('angry', 10.05), ('furious', 8.97),
    ('nostalgic', 10.33), ('caring', 8.60), ('sentimental', 8.46),
    ('confident', 10.74), ('hopeful', 8.74), ('anxious', 8.69),
    ('disgusted', 18.63), ('furious', 15.01), ('angry', 12.09),
    ('nostalgic', 17.90), ('surprised', 15.86), ('sentimental', 9.62),
    ('surprised', 13.90), ('content', 11.57), ('jealous', 11.08),
    ('confident', 14.26), ('hopeful', 12.41), ('excited', 10.62),
]
"""


# Splitting the data into categories and percentages
all_categories, all_percentages = zip(*all_clusters_data)

# Grouping data by cluster, taking 3 categories for each
all_clusters = [all_categories[i:i+3] for i in range(0, len(all_categories), 3)]

# Creating a matrix to represent the presence of each emotion in each cluster
all_emotion_clusters = pd.DataFrame(index=range(1, 33), columns=list(set(all_categories)))


# Creating a matrix to represent the presence of each emotion in each cluster
all_emotion_clusters = pd.DataFrame(index=range(1, 33), columns=list(set(all_categories)))

# Filling the matrix based on the presence of emotions in each cluster
for i, cluster in enumerate(all_clusters, start=1):
    for emotion in cluster:
        all_emotion_clusters.at[i, emotion] = 1

# Filling NaN values with 0
all_emotion_clusters.fillna(0, inplace=True)

# Sorting the columns alphabetically
all_emotion_clusters = all_emotion_clusters.sort_index(axis=1)


# Plotting the heatmap for all clusters
plt.figure(figsize=(20, 15))
sns.heatmap(all_emotion_clusters, cmap="YlGnBu", cbar=False)
plt.title("Emotion Presence in Clusters -- GPT-Generated Dialogue", fontsize=35)
plt.xlabel("Emotion", fontsize=25)
plt.ylabel("Cluster Category", fontsize=16)
plt.xticks(fontsize=16) # Set rotation angle to 45 degrees
plt.yticks(rotation=360, fontsize=20)

# Drawing grid lines
for i in range(all_emotion_clusters.shape[0] + 1):
    plt.axhline(i, lw=0.5, color='gray', linestyle='-')
for i in range(all_emotion_clusters.shape[1] + 1):
    plt.axvline(i, lw=0.5, color='gray', linestyle='-')


plt.savefig('GPT_clustersviz.pdf', format='pdf')
plt.show()
