# benchmarking.py

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance, ttest_ind
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

class Benchmarking:
    def __init__(self, X_full, y_full, X_core, y_core, X_compressed, y_compressed, X_test, y_test, feature_names, class_names):
        """
        Initializes the Benchmarking class with datasets and metadata.

        Parameters:
        - X_full: Full training feature set.
        - y_full: Full training labels.
        - X_core: Core training feature set.
        - y_core: Core training labels.
        - X_compressed: PCA-transformed core feature set.
        - y_compressed: Compressed core training labels.
        - X_test: Test feature set.
        - y_test: Test labels.
        - feature_names: List of feature names.
        - class_names: List of class names.
        """
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
        """
        Compares the performance of different models on the test set.

        Parameters:
        - rf_full: Model trained on the full dataset.
        - rf_core: Model trained on the core dataset.
        - rf_compressed: Model trained on the compressed core dataset.
        """
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

    def model_performance_summary(self, models_dict):
        """
        Prints a summary of model performance using cross-validation.

        Parameters:
        - models_dict: Dictionary of models with model names as keys.
        """
        print("\n=== Model Performance Summary with Cross-Validation ===")
        performance_data = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in models_dict.items():
            # Perform cross-validation and get scores
            cv_accuracy_scores = cross_val_score(model, self.X_test, self.y_test, cv=skf, scoring='accuracy', n_jobs=-1)
            cv_f1_scores = cross_val_score(model, self.X_test, self.y_test, cv=skf, scoring='f1_weighted', n_jobs=-1)
            
            # Calculate mean and standard deviation of scores
            mean_accuracy = cv_accuracy_scores.mean()
            std_accuracy = cv_accuracy_scores.std()
            mean_f1 = cv_f1_scores.mean()
            std_f1 = cv_f1_scores.std()

            # Store cross-validation performance metrics with detailed labels
            performance_data.append({
                'Model': name,
                'CV Accuracy (mean ± std)': f"{mean_accuracy:.4f} ± {std_accuracy:.4f}",
                'CV F1 Score (mean ± std)': f"{mean_f1:.4f} ± {std_f1:.4f}"
            })

        # Convert performance data to a DataFrame for clear output
        performance_df = pd.DataFrame(performance_data)
        print("\nCross-Validation Summary:")
        print(performance_df.to_string(index=False))  # Print DataFrame neatly

    def statistical_similarity_tests(self):
        """
        Performs statistical similarity tests between the full and core datasets.
        """
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

    def statistical_validation_of_compression(self):
        """
        Performs statistical validation of the compression method using KS tests and Chi-Square tests.
        """
        print("\n=== Statistical Validation of Compression ===")
        alpha = 0.05  # Significance level
        
        # 1. Kolmogorov-Smirnov tests on feature distributions
        print("\nPerforming Kolmogorov-Smirnov tests on feature distributions...")
        ks_pvalues = []
        for i in range(self.X_full.shape[1]):
            stat, p_value = ks_2samp(self.X_full[:, i], self.X_compressed[:, i])
            ks_pvalues.append(p_value)
        ks_pvalues = np.array(ks_pvalues)
        num_features = self.X_full.shape[1]
        num_similar = np.sum(ks_pvalues > alpha)
        print(f"Number of features with similar distributions: {num_similar}/{num_features}")
        print(f"Percentage: {num_similar / num_features * 100:.2f}%")
        
        # 2. Visualize feature distributions for selected features
        print("\nVisualizing feature distributions for selected features...")
        feature_indices = np.random.choice(range(num_features), size=5, replace=False)
        for idx in feature_indices:
            plt.figure(figsize=(8, 4))
            sns.kdeplot(self.X_full[:, idx], label='Full Data', shade=True)
            sns.kdeplot(self.X_compressed[:, idx], label='Compressed Data', shade=True)
            plt.title(f'Distribution Comparison for Feature {self.feature_names[idx]}')
            plt.legend()
            plt.show()
        
        # 3. Chi-Square test on class distributions
        print("\nComparing class distributions using Chi-Square test...")
        full_class_counts = pd.Series(self.y_full).value_counts().sort_index()
        compressed_class_counts = pd.Series(self.y_compressed).value_counts().sort_index()
        contingency_table = pd.DataFrame({
            'Full Data': full_class_counts,
            'Compressed Data': compressed_class_counts
        }).fillna(0)  # Handle any missing classes
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        print(f"Chi-Square Statistic: {chi2:.2f}, p-value: {p_value:.4f}")
        if p_value > alpha:
            print("Class distributions are similar.")
        else:
            print("Class distributions are significantly different.")

    def feature_importance_comparison(self, rf_full, rf_core):
        """
        Compares feature importances between full and core models.

        Parameters:
        - rf_full: Model trained on the full dataset.
        - rf_core: Model trained on the core dataset.
        """
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
        print(importance_df.sort_values('Difference', key=lambda x: x.abs(), ascending=False).head(10))

    def feature_similarity_logging(self, model_full, top_n=50):
        """
        Logs similarity measures for the top N important features.

        Parameters:
        - model_full: The full-trained model to extract feature importances.
        - top_n: Number of top features to log.
        """
        print("\n=== Detailed Feature Similarity Logging ===")
        
        # 1. Get Feature Importances from the Full Model
        print("\nCalculating feature importances from the full model...")
        feature_importances = model_full.feature_importances_
        
        # Create a DataFrame for feature importances
        feature_importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': feature_importances
        })
        
        # Sort features by importance
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        
        # Select top N important features
        top_features = feature_importance_df.head(top_n)['Feature'].tolist()
        top_feature_indices = [self.feature_names.index(f) for f in top_features]
        
        # 2. Log Similarity Measures for Top Features
        similarity_logs = []
        print(f"\nLogging similarity measures for top {top_n} important features...")
        for idx in top_feature_indices:
            feature_name = self.feature_names[idx]
            # Kolmogorov-Smirnov Test
            stat_ks, p_value_ks = ks_2samp(self.X_full[:, idx], self.X_compressed[:, idx])
            # Jensen-Shannon Divergence
            hist_full, bin_edges = np.histogram(self.X_full[:, idx], bins=50, density=True)
            hist_compressed, _ = np.histogram(self.X_compressed[:, idx], bins=bin_edges, density=True)
            hist_full += 1e-8
            hist_compressed += 1e-8
            js_distance = jensenshannon(hist_full, hist_compressed)
            # Wasserstein Distance
            wasserstein_dist = wasserstein_distance(self.X_full[:, idx], self.X_compressed[:, idx])
            # Append to logs
            similarity_logs.append({
                'Feature': feature_name,
                'KS Statistic': stat_ks,
                'KS p-value': p_value_ks,
                'Jensen-Shannon Distance': js_distance,
                'Wasserstein Distance': wasserstein_dist
            })
        
        # Convert logs to DataFrame
        similarity_logs_df = pd.DataFrame(similarity_logs)
        
        # Save logs to a CSV file
        similarity_logs_df.to_csv('feature_similarity_logs.csv', index=False)
        print("\nFeature similarity logs saved to 'feature_similarity_logs.csv'.")
        
        # 3. Display the logs
        print("\nTop Features Similarity Measures:")
        print(similarity_logs_df.head(10))  # Display top 10 for brevity
        
        # 4. Visualize Distributions of Top Features
        print("\nVisualizing distributions of top important features...")
        for idx in top_feature_indices[:5]:  # Visualize top 5 features
            feature_name = self.feature_names[idx]
            plt.figure(figsize=(8, 4))
            sns.kdeplot(self.X_full[:, idx], label='Full Data', shade=True)
            sns.kdeplot(self.X_compressed[:, idx], label='Compressed Data', shade=True)
            plt.title(f'Distribution Comparison for Feature: {feature_name}')
            plt.legend()
            plt.show()

    def visualize_feature_distributions(self, feature_indices=None, datasets=['full', 'core', 'compressed']):
        """
        Visualizes the distributions of specified features across different datasets.

        Parameters:
        - feature_indices: List of feature indices to visualize. If None, selects first 5 features.
        - datasets: List of datasets to include in the plots. Options: 'full', 'core', 'compressed'.
        """
        print("\n=== Visualizing Feature Distributions ===")
        if feature_indices is None:
            num_features = min(5, self.X_full.shape[1])
            feature_indices = range(num_features)
        for idx in feature_indices:
            plt.figure(figsize=(10, 4))
            if 'full' in datasets:
                sns.kdeplot(self.X_full[:, idx], label='Full Dataset', shade=True)
            if 'core' in datasets:
                sns.kdeplot(self.X_core[:, idx], label='Core Dataset', shade=True)
            if 'compressed' in datasets:
                sns.kdeplot(self.X_compressed[:, idx], label='Compressed Dataset', shade=True)
            plt.title(f'Distribution of Feature: {self.feature_names[idx]}')
            plt.legend()
            plt.show()

    def pca_visualization(self):
        """
        Visualizes the PCA-transformed data comparing full and compressed datasets.
        """
        print("\n=== PCA Visualization ===")
        plt.figure(figsize=(10, 8))
        plt.scatter(self.X_full[:, 0], self.X_full[:, 1], c=self.y_full, cmap='viridis', alpha=0.1, label='Full Data')
        plt.scatter(self.X_compressed[:, 0], self.X_compressed[:, 1], c=self.y_compressed, cmap='coolwarm', edgecolor='k', label='Compressed Data')
        plt.title('PCA Visualization of Full Data vs. Compressed Data')
        plt.legend()
        plt.show()

    def tsne_visualization(self):
        """
        Performs t-SNE visualization for full and core datasets.
        """
        print("\n=== t-SNE Visualization ===")
        tsne = TSNE(n_components=2, random_state=42)
        X_full_sample = self.X_full[:1000] if self.X_full.shape[0] > 1000 else self.X_full
        y_full_sample = self.y_full[:1000] if self.y_full.shape[0] > 1000 else self.y_full
        X_full_2d = tsne.fit_transform(X_full_sample)
        X_core_2d = tsne.fit_transform(self.X_core)
        
        plt.figure(figsize=(8, 6))
        scatter_full = plt.scatter(X_full_2d[:, 0], X_full_2d[:, 1], c=y_full_sample, cmap='viridis', alpha=0.5, label='Full Dataset')
        scatter_core = plt.scatter(X_core_2d[:, 0], X_core_2d[:, 1], c=self.y_core, cmap='coolwarm', alpha=0.8, label='Core Dataset')
        plt.title('t-SNE Visualization of Datasets')
        plt.legend()
        plt.show()

    def clustering_evaluation(self):
        """
        Evaluates clustering performance on the core dataset using Silhouette Score and Davies-Bouldin Index.
        """
        print("\n=== Clustering Evaluation Metrics ===")
        # Evaluate clustering on the core dataset
        kmeans = MiniBatchKMeans(n_clusters=10, random_state=42)
        cluster_labels = kmeans.fit_predict(self.X_core)
        silhouette_avg = silhouette_score(self.X_core, cluster_labels)
        db_index = davies_bouldin_score(self.X_core, cluster_labels)
        print(f"Silhouette Score on Core Dataset: {silhouette_avg:.4f}")
        print(f"Davies-Bouldin Index on Core Dataset: {db_index:.4f}")

    def cross_validation_checks(self, model):
        """
        Performs cross-validation checks on the core dataset.

        Parameters:
        - model: The model to evaluate.
        """
        print("\n=== Cross-Validation Checks ===")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, self.X_core, self.y_core, cv=skf, scoring='accuracy', n_jobs=-1)
        print(f"Cross-Validation Accuracy Scores: {cv_scores}")
        print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    def check_data_leakage(self):
        """
        Checks for data leakage by ensuring no overlap between training and test sets.
        """
        print("\n=== Data Leakage Check ===")
        # Convert datasets to structured arrays for comparison
        X_core_structured = self.X_core.view([('', self.X_core.dtype)] * self.X_core.shape[1])
        X_test_structured = self.X_test.view([('', self.X_test.dtype)] * self.X_test.shape[1])
        overlap = np.intersect1d(X_core_structured, X_test_structured)
        print(f"Number of overlapping samples: {len(overlap)}")
        if overlap.size == 0:
            print("No data leakage detected: No overlap between training and test sets.")
        else:
            print("Warning: Potential data leakage detected!")

    def plot_roc_curves(self, models_dict):
        """
        Plots ROC curves for the provided models.

        Parameters:
        - models_dict: Dictionary of models with model names as keys.
        """
        print("\n=== ROC Curve Comparison ===")
        plt.figure(figsize=(10, 8))
        for name, model in models_dict.items():
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(self.X_test)
                if len(self.class_names) == 2:
                    # Binary classification
                    fpr, tpr, _ = roc_curve(self.y_test, y_score[:, 1])
                    auc_score = roc_auc_score(self.y_test, y_score[:, 1])
                    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.2f})')
                else:
                    # Multi-class classification using One-vs-Rest
                    for i in range(len(self.class_names)):
                        fpr, tpr, _ = roc_curve(self.y_test == i, y_score[:, i])
                        auc_score = roc_auc_score(self.y_test == i, y_score[:, i])
                        plt.plot(fpr, tpr, label=f'{name} Class {self.class_names[i]} (AUC = {auc_score:.2f})')
            else:
                print(f"Model {name} does not support probability predictions.")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title('ROC Curves for Different Models')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()

    def plot_feature_distribution(self, feature_idx, datasets=['full', 'compressed'], title=None):
        """
        Plots the distribution of a single feature across specified datasets.

        Parameters:
        - feature_idx: Index of the feature to plot.
        - datasets: List of datasets to include in the plot. Options: 'full', 'compressed'.
        - title: Title of the plot.
        """
        plt.figure(figsize=(8, 4))
        if 'full' in datasets:
            sns.kdeplot(self.X_full[:, feature_idx], label='Full Data', shade=True)
        if 'compressed' in datasets:
            sns.kdeplot(self.X_compressed[:, feature_idx], label='Compressed Data', shade=True)
        plt.title(title or f'Distribution for Feature {self.feature_names[feature_idx]}')
        plt.legend()
        plt.show()

    def plot_feature_importances(self, model, top_n=20):
        """
        Plots the top N feature importances from a given model.

        Parameters:
        - model: The trained model with feature_importances_ attribute.
        - top_n: Number of top features to plot.
        """
        if not hasattr(model, 'feature_importances_'):
            print("The provided model does not have feature_importances_ attribute.")
            return
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [self.feature_names[i] for i in indices]
        top_importances = importances[indices]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_importances, y=top_features, palette='viridis')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.show()
