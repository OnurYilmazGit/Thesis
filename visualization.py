import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandas.plotting import lag_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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
