# benchmarking.py

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.stats import ks_2samp, ttest_ind
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
from sklearn.cluster import MiniBatchKMeans

warnings.filterwarnings("ignore")

class Benchmarking:
    def __init__(self, X_full, y_full, X_core, y_core, X_compressed, y_compressed, X_test, y_test, feature_names, class_names):
        self.X_full = X_full
        self.y_full = y_full
        self.X_core = X_core
        self.y_core = y_core
        self.X_compressed = X_compressed
        self.y_compressed = y_compressed
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.class_names = class_names

    def compare_model_performance(self, rf_full, rf_core, rf_compressed):
        print("\n=== Model Performance Comparison ===")
        models = {
            'Full Dataset Model': rf_full,
            'Core Dataset Model': rf_core,
            'Compressed Core Model': rf_compressed
        }
        for name, model in models.items():
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            print(f"\n{name}:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print("Classification Report:")
            print(classification_report(self.y_test, y_pred, target_names=self.class_names, zero_division=0))
            print("Confusion Matrix:")
            print(confusion_matrix(self.y_test, y_pred))

    def statistical_similarity_tests(self):
        print("\n=== Statistical Similarity Tests ===")
        p_values = []
        for i in range(self.X_full.shape[1]):
            feature_full = self.X_full[:, i]
            feature_core = self.X_core[:, i]
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = ks_2samp(feature_full, feature_core)
            # t-test
            t_stat, t_p = ttest_ind(feature_full, feature_core, equal_var=False)
            p_values.append({'Feature': self.feature_names[i], 'KS_p_value': ks_p, 't_p_value': t_p})
        p_values_df = pd.DataFrame(p_values)
        print("P-values from statistical tests between full and core datasets:")
        print(p_values_df.head())
        # You can decide on a threshold to determine similarity

    def feature_importance_comparison(self, rf_full, rf_core):
        print("\n=== Feature Importance Comparison ===")
        importances_full = rf_full.feature_importances_
        importances_core = rf_core.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Full Importance': importances_full,
            'Core Importance': importances_core
        })
        importance_df['Difference'] = importance_df['Full Importance'] - importance_df['Core Importance']
        print("Top 10 features with the largest difference in importance:")
        print(importance_df.sort_values('Difference', key=abs, ascending=False).head(10))

    def visualize_feature_distributions(self):
        print("\n=== Visualizing Feature Distributions ===")
        num_features = min(5, self.X_full.shape[1])  # Limit to first 5 features for visualization
        
        plt.figure(figsize=(12, 8))  # Single figure for all feature plots
        
        # Loop through the features and plot them on the same graph
        for i in range(num_features):
            sns.kdeplot(self.X_full[:, i], label=f'Full Dataset: {self.feature_names[i]}', shade=True, linewidth=5
                        )
            sns.kdeplot(self.X_core[:, i], label=f'Core Dataset: {self.feature_names[i]}', shade=True, linewidth=5)
        
        # Adding title, labels and legend
        plt.title('Distributions of Top 5 Features')
        plt.xlabel('Feature Value')
        plt.ylabel('Density')
        plt.legend(loc='best')
        plt.show()

    def clustering_evaluation(self):
        print("\n=== Clustering Evaluation Metrics ===")
        # Evaluate clustering on the core dataset
        kmeans = MiniBatchKMeans(n_clusters=10, random_state=42)
        cluster_labels = kmeans.fit_predict(self.X_core)
        silhouette_avg = silhouette_score(self.X_core, cluster_labels)
        db_index = davies_bouldin_score(self.X_core, cluster_labels)
        print(f"Silhouette Score on Core Dataset: {silhouette_avg:.4f}")
        print(f"Davies-Bouldin Index on Core Dataset: {db_index:.4f}")

    def cross_validation_checks(self, model):
        print("\n=== Cross-Validation Checks ===")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, self.X_core, self.y_core, cv=skf, scoring='accuracy', n_jobs=-1)
        print(f"Cross-Validation Accuracy Scores: {cv_scores}")
        print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    def check_data_leakage(self):
        print("\n=== Data Leakage Check ===")
        # Ensure that there is no overlap between training and test sets
        overlap = np.intersect1d(self.X_core, self.X_test)
        print(len(overlap))
        print(overlap.size)
        print("Size of overlap", overlap)
        if overlap.size == 0:
            print("No data leakage detected: No overlap between training and test sets.")
        else:
            print("Warning: Potential data leakage detected!")

    def tsne_visualization(self):
        print("\n=== t-SNE Visualization ===")
        tsne = TSNE(n_components=2, random_state=42)
        X_full_2d = tsne.fit_transform(self.X_full[:1000])  # Limit to 1000 samples for performance
        X_core_2d = tsne.fit_transform(self.X_core)
        plt.figure(figsize=(8, 6))
        plt.scatter(X_full_2d[:, 0], X_full_2d[:, 1], c=self.y_full[:1000], cmap='viridis', alpha=0.5, label='Full Dataset')
        plt.scatter(X_core_2d[:, 0], X_core_2d[:, 1], c=self.y_core, cmap='coolwarm', alpha=0.8, label='Core Dataset')
        plt.title('t-SNE Visualization of Datasets')
        plt.legend()
        plt.show()
