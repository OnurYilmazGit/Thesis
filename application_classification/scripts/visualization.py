import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandas.plotting import lag_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go


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

    # New methods for time series analysis

    def plot_time_series(self, data, title='Time Series Data'):
        """Plot time series data to visualize trends over time."""
        self._initialize_plot(title, xlabel='Time', ylabel='Value')
        plt.plot(data['Timestamp'], data['Value'])
        plt.show()

    def plot_lag(self, data, lag=1, title='Lag Plot'):
        """Plot lagged values of the time series to check for autocorrelation."""
        self._initialize_plot(f'Lag Plot (Lag={lag})', xlabel=f'Value at t', ylabel=f'Value at t-{lag}')
        lag_plot(data['Value'], lag=lag)
        plt.show()


    def plot_acf(self, data, lags=50, title='Autocorrelation Function'):
        """Plot the autocorrelation function of the time series."""
        self._initialize_plot(title, xlabel='Lag', ylabel='Autocorrelation')
        plot_acf(data['Value'], lags=lags)
        plt.show()

    def plot_pacf(self, data, lags=50, title='Partial Autocorrelation Function'):
        """Plot the partial autocorrelation function of the time series."""
        self._initialize_plot(title, xlabel='Lag', ylabel='Partial Autocorrelation')
        plot_pacf(data['Value'], lags=lags)
        plt.show()

    def plot_pca_variance(self, pca):
        """Plot cumulative variance explained by PCA components."""
        self._initialize_plot('Cumulative Explained Variance by PCA Components', xlabel='Components', ylabel='Cumulative Explained Variance')
        plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
        plt.show()

    def plot_elbow_method(self, K, inertia):
        """Plot the elbow method to select the number of clusters."""
        self._initialize_plot('Elbow Method for Optimal K', xlabel='Number of clusters', ylabel='Inertia')
        plt.plot(K, inertia, marker='o')
        plt.show()

    def plot_clusters(self, X_pca, labels):
        """Plot K-Means clustering results."""
        self._initialize_plot('KMeans Clustering Results', xlabel='PCA Component 1', ylabel='PCA Component 2')
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='viridis', alpha=0.7)
        plt.show()

    def plot_centroids(self, centroids):
        """Plot centroids of clusters."""
        self._initialize_plot('Cluster Centroids', xlabel='PCA Component 1', ylabel='PCA Component 2')
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='x')
        plt.show()

    def plot_interactive_time_series(self, data):
        """Create an interactive time series plot using Plotly."""
        fig = px.line(data, x='Time', y='Value', title='Interactive Time Series Plot')
        fig.update_layout(xaxis_title='Time', yaxis_title='Value')
        fig.show()

    def plot_interactive_acf(self, data, lags=50):
        """Create an interactive Autocorrelation Function plot using Plotly."""
        acf_values = plot_acf(data['Value'], lags=lags, alpha=0.05, fft=False)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=acf_values.lags, y=acf_values.acf, marker=dict(color="blue")))
        fig.update_layout(title="Interactive Autocorrelation", xaxis_title="Lag", yaxis_title="ACF Value")
        fig.show()

    def plot_interactive_pacf(self, data, lags=50):
        """Create an interactive Partial Autocorrelation Function plot using Plotly."""
        pacf_values = plot_pacf(data['Value'], lags=lags, alpha=0.05)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=pacf_values.lags, y=pacf_values.pacf, marker=dict(color="green")))
        fig.update_layout(title="Interactive Partial Autocorrelation", xaxis_title="Lag", yaxis_title="PACF Value")
        fig.show()

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

    def plot_differenced_series(self, time_series, title="Differenced Time Series"):
        """Plot the differenced time series."""
        self._initialize_plot(title, xlabel='Time', ylabel='Differenced Value')
        plt.plot(time_series, color='blue')
        plt.show()

    def plot_seasonal_decomposition(self, decomposition):
        """Plot seasonal decomposition components."""
        decomposition.plot()
        plt.show()

    def plot_lag(self, data, lag=1, title='Lag Plot'):
        """Plot lagged values of the time series to check for autocorrelation."""
        self._initialize_plot(f'Lag Plot (Lag={lag})', xlabel=f'Value at t', ylabel=f'Value at t-{lag}')
        lag_plot(data['Value'], lag=lag)
        plt.show()

    def plot_pca_variance(self, pca):
        """Plot cumulative variance explained by PCA components."""
        self._initialize_plot('Cumulative Explained Variance by PCA Components', xlabel='Components', ylabel='Cumulative Explained Variance')
        plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
        plt.show()

    def plot_dimension_reduction(self, X_before, X_after, title="Dimension Reduction"):
        """Plot scatter plots before and after PCA for dimensionality reduction."""
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
        """Plot K-Means clustering results."""
        self._initialize_plot('KMeans Clustering Results', xlabel='PCA Component 1', ylabel='PCA Component 2')
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='viridis', alpha=0.7)
        plt.show()

    def plot_centroids(self, centroids):
        """Plot centroids of clusters."""
        self._initialize_plot('Cluster Centroids', xlabel='PCA Component 1', ylabel='PCA Component 2')
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='x')
        plt.show()

    def plot_elbow_method(self, K, inertia):
        """Plot the elbow method to select the number of clusters."""
        self._initialize_plot('Elbow Method for Optimal K', xlabel='Number of clusters', ylabel='Inertia')
        plt.plot(K, inertia, marker='o')
        plt.show()

    def plot_seasonal_decomposition(self, time_series, frequency=12, title="Seasonality, Trend, and Noise"):
        """Plot seasonal decomposition of time series."""
        decomposition = seasonal_decompose(time_series, period=frequency)
        decomposition.plot()
        plt.suptitle(title)
        plt.show()

    def plot_autocorrelation(self, data, lags=50, title="Autocorrelation"):
        """Plot autocorrelation for the given data."""
        plt.figure(figsize=(10, 6))
        plot_acf(data, lags=lags)
        plt.title(title)
        plt.show()

    def plot_partial_autocorrelation(self, data, lags=50, title="Partial Autocorrelation"):
        """Plot partial autocorrelation for the given data."""
        plt.figure(figsize=(10, 6))
        plot_pacf(data, lags=lags)
        plt.title(title)
        plt.show()

    

    def plot_tsne(X_full, X_core, y_full, y_core, class_names, filename):
        """Function to plot t-SNE visualization comparing full and core datasets."""
        print("\n=== Step 8: t-SNE Visualization ===")
        
        # Initialize t-SNE for 2D visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        
        # Fit t-SNE on full and core datasets
        X_full_tsne = tsne.fit_transform(X_full)
        X_core_tsne = tsne.fit_transform(X_core)
        
        # Create a figure for t-SNE comparison
        plt.figure(figsize=(10, 8))

        # Plot t-SNE for full dataset
        plt.scatter(X_full_tsne[:, 0], X_full_tsne[:, 1], c=y_full, cmap='tab10', alpha=0.4, label="Full Dataset", s=10)

        # Overlay t-SNE for core dataset
        plt.scatter(X_core_tsne[:, 0], X_core_tsne[:, 1], c=y_core, cmap='tab10', edgecolor='black', label="Core Dataset", s=40)
        
        # Add title and legend
        plt.title("t-SNE Visualization of Full and Core Datasets")
        plt.legend(loc='best')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        
        # Save the figure
        plt.savefig(filename)
        plt.close()

        print(f"t-SNE plot saved as {filename}")


    def visualize_tsne_with_pca(X, y, n_components_pca=30, n_components_tsne=2, perplexity=30, n_iter=500):
        """
        Visualizes the dataset using PCA + t-SNE.
        
        Args:
            X: The feature matrix.
            y: The target labels.
            n_components_pca: Number of components for PCA.
            n_components_tsne: Number of dimensions for t-SNE (default is 2 for 2D visualization).
            perplexity: Controls the number of neighbors used in t-SNE.
            n_iter: Number of iterations for optimization.
        """
        print("\n=== Applying PCA ===")
        # Apply PCA to reduce dimensionality
        pca = PCA(n_components=n_components_pca)
        X_pca = pca.fit_transform(X)
        print(f"Reduced dimensions from {X.shape[1]} to {n_components_pca} using PCA.")
        
        print("\n=== Visualizing t-SNE after PCA ===")
        # Apply t-SNE on PCA-reduced data
        tsne = TSNE(n_components=n_components_tsne, perplexity=perplexity, n_iter=n_iter, n_jobs=-1)
        X_tsne = tsne.fit_transform(X_pca)

        # Plot the t-SNE results
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette="deep", legend="full", alpha=0.7)
        plt.title(f"t-SNE Visualization after PCA ({n_components_pca} components)")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("tsne_with_pca_visualization.png")
        plt.show()

    def export_for_tsne(X_full, y_full, X_core, y_core, X_stratified_core, y_stratified_core, folder='graph_output'):
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        # Export full dataset
        full_data = pd.DataFrame(X_full)
        full_data['label'] = y_full.values
        full_data.to_csv(os.path.join(folder, 'full_dataset.csv'), index=False)
        
        # Export core dataset
        core_data = pd.DataFrame(X_core)
        core_data['label'] = y_core.values
        core_data.to_csv(os.path.join(folder, 'core_dataset.csv'), index=False)
        
        # Export stratified sampled core dataset
        stratified_core_data = pd.DataFrame(X_stratified_core)
        stratified_core_data['label'] = y_stratified_core.values
        stratified_core_data.to_csv(os.path.join(folder, 'stratified_core_dataset.csv'), index=False)
        
        print(f"Datasets exported to {folder} folder for t-SNE generation.")

