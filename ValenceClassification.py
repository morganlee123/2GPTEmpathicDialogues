# Runs the valence classification experiment. Specifics detailed in the paper: https://arxiv.org/pdf/2401.16587.pdf
# Code author: Morgan Sandler (sandle20@msu.edu)
# Note: Code is currently set up to perform valence classification for GPT-generated dialogues, simple modifications
# of the TODO's listed will allow you to reuse for human-generated.

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, f1_score
import os
from tqdm import tqdm

# Create the directory if it doesn't exist
# TODO: Change this for human_dialogues if reusing for that.
if not os.path.exists('ValExperiment/gpt_dialogues'):
    os.makedirs('ValExperiment/gpt_dialogues')

# Function to plot confusion matrix
def plot_confusion_matrix(conf_matrix, label_names, model_name):
    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names,
                yticklabels=label_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'{model_name} Confusion Matrix')
    plt.savefig(f'ValExperiment/gpt_dialogues/{model_name}_conf_matrix.png')

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

# Load Data and Labels
# TODO: Change this for human_dialogues embeddings if reusing for that.
with open('2GPTEmpathicDialoguesAsEmbeddings.pkl', 'rb') as f:
    df = pickle.load(f)

embeddings = np.vstack(df.embedding)
merged_labels = merge_categories(df.context)  # Ensure this function is defined
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(merged_labels)

# 5-Fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Parameter Grids for GridSearch
param_grid_rf = {'n_estimators': [50, 100, 200]}
param_grid_svm = {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1]}
param_grid_mlp = {
    'hidden_layer_sizes': [(100,), (300, 200, 100), (150, 150)],
    'max_iter': [300, 500]
}

# Results container for each fold
results = []

for train_index, test_index in tqdm(skf.split(embeddings, labels)):
    X_train, X_test = embeddings[train_index], embeddings[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # RandomForest Classifier with GridSearch
    rf_classifier = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=3)
    rf_classifier.fit(X_train, y_train)
    y_pred_rf = rf_classifier.predict(X_test)
    rf_conf_matrix = confusion_matrix(y_test, y_pred_rf)
    rf_f1 = f1_score(y_test, y_pred_rf, average='weighted')
    results.append(['Random Forest', rf_conf_matrix, rf_f1])

    # SVM with RBF Kernel with GridSearch
    svm_classifier = GridSearchCV(SVC(kernel='rbf'), param_grid_svm, cv=3)
    svm_classifier.fit(X_train, y_train)
    y_pred_svm = svm_classifier.predict(X_test)
    svm_conf_matrix = confusion_matrix(y_test, y_pred_svm)
    svm_f1 = f1_score(y_test, y_pred_svm, average='weighted')
    results.append(['SVM with RBF Kernel', svm_conf_matrix, svm_f1])

    # Four-layer Neural Network using MLPClassifier with GridSearch
    mlp_classifier = GridSearchCV(MLPClassifier(), param_grid_mlp, cv=3)
    mlp_classifier.fit(X_train, y_train)
    y_pred_mlp = mlp_classifier.predict(X_test)
    mlp_conf_matrix = confusion_matrix(y_test, y_pred_mlp)
    mlp_f1 = f1_score(y_test, y_pred_mlp, average='weighted')
    results.append(['Four-layer MLP', mlp_conf_matrix, mlp_f1])

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes, title, filename):
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Normalized Confusion Matrix for {title}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename)

# Results aggregation
rf_conf_matrices = []
svm_conf_matrices = []
mlp_conf_matrices = []
rf_f1_scores = []
svm_f1_scores = []
mlp_f1_scores = []

for result in results:
    if result[0] == 'Random Forest':
        rf_conf_matrices.append(result[1])
        rf_f1_scores.append(result[2])
    elif result[0] == 'SVM with RBF Kernel':
        svm_conf_matrices.append(result[1])
        svm_f1_scores.append(result[2])
    elif result[0] == 'Four-layer MLP':
        mlp_conf_matrices.append(result[1])
        mlp_f1_scores.append(result[2])

# Averaging confusion matrices and F1 scores
avg_rf_conf_matrix = np.mean(rf_conf_matrices, axis=0)
avg_svm_conf_matrix = np.mean(svm_conf_matrices, axis=0)
avg_mlp_conf_matrix = np.mean(mlp_conf_matrices, axis=0)
avg_rf_f1 = np.mean(rf_f1_scores)
avg_svm_f1 = np.mean(svm_f1_scores)
avg_mlp_f1 = np.mean(mlp_f1_scores)

# Calculating standard deviations
std_rf_conf_matrix = np.std(rf_conf_matrices, axis=0)
std_svm_conf_matrix = np.std(svm_conf_matrices, axis=0)
std_mlp_conf_matrix = np.std(mlp_conf_matrices, axis=0)
std_rf_f1 = np.std(rf_f1_scores)
std_svm_f1 = np.std(svm_f1_scores)
std_mlp_f1 = np.std(mlp_f1_scores)

# Plot and save confusion matrices
# TODO: Change this for human_dialogues folder if reusing for that.
original_label_names = label_encoder.inverse_transform(np.unique(labels))
plot_confusion_matrix(avg_rf_conf_matrix, original_label_names, 'Random Forest', 'ValExperiment/gpt_dialogues/Random_Forest_CM.png')
plot_confusion_matrix(avg_svm_conf_matrix, original_label_names, 'SVM RBF', 'ValExperiment/gpt_dialogues/SVM_RBF_CM.png')
plot_confusion_matrix(avg_mlp_conf_matrix, original_label_names, 'Four-layer MLP', 'ValExperiment/gpt_dialogues/Four_layer_MLP_CM.png')

# Create averaged results DataFrame
averaged_results = [
            ['Random Forest', avg_rf_f1, std_rf_f1],
                ['SVM with RBF Kernel', avg_svm_f1, std_svm_f1],
                    ['Four-layer MLP', avg_mlp_f1, std_mlp_f1]

        ]
averaged_results_df = pd.DataFrame(averaged_results, columns=['Classifier', 'Average Weighted F1-Score', 'Standard Deviation F1-Score'])
# TODO: Change this for human_dialogues folder if reusing for that.
averaged_results_df.to_csv('ValExperiment/gpt_dialogues/averaged_valclass_results.csv')

# Save the six lists as separate files
# TODO: Change this for human_dialogues folder if reusing for that.
np.save('ValExperiment/gpt_dialogues/rf_conf_matrices.npy', rf_conf_matrices)
np.save('ValExperiment/gpt_dialogues/svm_conf_matrices.npy', svm_conf_matrices)
np.save('ValExperiment/gpt_dialogues/mlp_conf_matrices.npy', mlp_conf_matrices)
np.save('ValExperiment/gpt_dialogues/rf_f1_scores.npy', rf_f1_scores)
np.save('ValExperiment/gpt_dialogues/svm_f1_scores.npy', svm_f1_scores)
np.save('ValExperiment/gpt_dialogues/mlp_f1_scores.npy', mlp_f1_scores)


