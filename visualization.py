import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Visualization:
    def __init__(self):
        self.default_figsize = (10, 6)

    def _initialize_plot(self, title, xlabel='', ylabel='', grid=True):
        plt.figure(figsize=self.default_figsize)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if grid:
            plt.grid(True)
    
    def plot_scaled_features(self, X_scaled, feature_columns):
        self._initialize_plot('Distribution of Scaled Features')
        sns.boxplot(data=X_scaled)
        plt.xticks(ticks=np.arange(len(feature_columns)), labels=feature_columns, rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def plot_pca_variance(self, pca):
        self._initialize_plot('Cumulative Explained Variance by PCA Components', xlabel='Number of Components', ylabel='Cumulative Explained Variance')
        plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
        plt.show()

    def plot_dimension_reduction(self, X_before, X_after, title="Dimension Reduction"):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(X_before[:, 0], X_before[:, 1], alpha=0.5)
        plt.title(f'{title} - Before')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.scatter(X_after[:, 0], X_after[:, 1], alpha=0.5)
        plt.title(f'{title} - After')
        plt.grid(True)
        plt.show()

    def plot_clusters(self, X_pca, labels):
        self._initialize_plot('KMeans Clustering Results', xlabel='PCA Component 1', ylabel='PCA Component 2')
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='viridis', alpha=0.7)
        plt.show()

    def plot_elbow_method(self, K, inertia):
        self._initialize_plot('Elbow Method for Optimal K', xlabel='Number of clusters', ylabel='Inertia (Within-Cluster Sum of Squares)')
        plt.plot(K, inertia, marker='o')
        plt.show()

    def plot_core_set(self, core_set_pca, core_set_labels):
        self._initialize_plot('Core Set Visualization', xlabel='PCA Component 1', ylabel='PCA Component 2')
        sns.scatterplot(x=core_set_pca[:, 0], y=core_set_pca[:, 1], hue=core_set_labels, palette='viridis', alpha=0.7)
        plt.show()

    def plot_distribution(self, df, title="Feature Distribution"):
        df.hist(bins=30, figsize=(12, 8))
        plt.suptitle(title)
        plt.show()

    def plot_rolling_statistics(self, original, rolling_mean, rolling_std, window_size):
        self._initialize_plot('Rolling Mean & Standard Deviation')
        plt.plot(original, color='blue', label='Original')
        plt.plot(rolling_mean, color='red', label=f'Rolling Mean (window={window_size})')
        plt.fill_between(range(len(original)), 
                         rolling_mean - rolling_std, 
                         rolling_mean + rolling_std, 
                         color='gray', alpha=0.2, label='Rolling Std (±1)')
        plt.legend(loc='best')
        plt.show()

    def plot_difference(self, original, difference):
        self._initialize_plot('Original Value and Rate of Change')
        plt.plot(original, color='blue', label='Original Value')
        plt.plot(difference, color='orange', label='Difference (Rate of Change)')
        plt.legend(loc='best')
        plt.show()

    def plot_time_based_features(self, data):
        self._initialize_plot('Distribution of Minute-based Features', xlabel='Minute', ylabel='Frequency')
        sns.histplot(data['minute'], kde=False)
        plt.show()

    def plot_correlation_heatmap(self, df):
        """Plot a heatmap of the correlation matrix between features."""
        corr = df.corr()
        self._initialize_plot('Correlation Heatmap', xlabel='Features', ylabel='Features')
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
        plt.show()

    def plot_feature_importances(self, feature_importances, feature_columns):
        """Plot the feature importances from a model like Random Forest."""
        self._initialize_plot('Feature Importances', xlabel='Importance', ylabel='Features')
        sns.barplot(x=feature_importances, y=feature_columns)
        plt.xticks(rotation=45, ha='right')
        plt.show()
