import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from joblib import Parallel, delayed
import logging
from visualization import Visualization

class DataPreprocessor:
    def __init__(self, data, window_size=10, use_incremental_pca=False):
        self.data = data
        self.window_size = window_size
        self.use_incremental_pca = use_incremental_pca
        self.visualization = Visualization()
        self.scaler = StandardScaler()

    def feature_engineering(self):
        """Create features such as lagged values and differences based on seconds."""
        logging.info("Starting feature engineering")
        
        # Log the initial shape of the data
        print(f"Initial data shape: {self.data.shape}")

        # Create time-based features focused on seconds
        time_col = pd.to_datetime(self.data['Time'], unit='s')
        self.data['second'] = time_col.dt.second


        # Lag features to capture past values
        for lag in range(1, 10):  # Create lag features for 1 to 10 seconds
            self.data[f'lag_{lag}'] = self.data['Value'].shift(lag)
            #print(f"After creating lag_{lag}, data shape: {self.data.shape}")

        # Difference between consecutive values to capture changes
        for lag in range(1, 10):  # Create diff features for 1 to 10 seconds
            self.data[f'diff_{lag}'] = self.data['Value'].diff(lag)
            #print(f"After creating diff_{lag}, data shape: {self.data.shape}")
        
        # Option 2: Use forward fill/backward fill (if applicable)
        self.data.fillna(method='ffill', inplace=True)
        self.data.fillna(method='bfill', inplace=True)

        # Log and store data snapshot if needed
        if self.data.empty:
            self.data.to_csv("empty_data_snapshot.csv", index=False)
            raise ValueError("Data is empty after feature engineering. Check data processing steps.")

        # Save lagged and difference features to a new CSV file
        self.data.to_csv('data_with_lag_diff.csv', index=False)
        logging.info("Feature engineering completed")
        return self.data

    def normalize_data(self, feature_columns):
        """Scale the data using StandardScaler."""
        logging.info("Normalizing data")
        X = self.data[feature_columns].to_numpy(dtype=np.float32)
        
        # Parallelize the scaling process (Note: StandardScaler is already highly optimized and parallelized internally,
        # but for demonstration purposes, you could wrap this in a parallel operation if needed)
        X_scaled = Parallel(n_jobs=-1)(delayed(self.scaler.fit_transform)(X_chunk) for X_chunk in np.array_split(X, 4))

        # Concatenate back into a single array
        X_scaled = np.concatenate(X_scaled, axis=0)
        
        # Export X_scaled as csv for debugging
        #pd.DataFrame(X_scaled).to_csv('X_scaled.csv', index=False)
        return X_scaled


    def apply_pca(self, X_scaled, n_components=None):
        """Apply PCA to the scaled data."""
        logging.info("Applying PCA")
        if self.use_incremental_pca:
            pca = IncrementalPCA(n_components=n_components, batch_size=500)
        else:
            pca = PCA(n_components=n_components)

        X_pca = pca.fit_transform(X_scaled)

        self.visualization.plot_pca_variance(pca)
        return X_pca, pca

    def optimize_cluster_selection(self, X_pca, max_clusters=200):
        """Determine the optimal number of clusters using the elbow method."""
        logging.info("Optimizing cluster selection using the elbow method")
        inertia = []
        K = range(2, max_clusters + 1)
        for k in K:
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=200)
            kmeans.fit(X_pca)
            inertia.append(kmeans.inertia_)
        
        return self.find_elbow_point(K, inertia)

    @staticmethod
    def find_elbow_point(K, inertia):
        """Identify the elbow point in the inertia plot."""
        derivatives = np.diff(inertia)
        second_derivatives = np.diff(derivatives)
        elbow_index = np.argmax(second_derivatives) + 2
        return K[elbow_index]
        
    def refine_cluster_selection(self, X_pca, n_clusters=500, points_per_cluster=20):
        """Refine the cluster selection by choosing representative points."""
        logging.info(f"Refining cluster selection with {n_clusters} clusters")

        # Apply KMeans clustering
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
        self.data['Cluster'] = kmeans.fit_predict(X_pca)

        # Ensure 'Time' is parsed as a full datetime object (with hour, minute, second precision)
        self.data['Time'] = pd.to_datetime(self.data['Time'], format='%Y-%m-%d %H:%M:%S')

        # Select core set points from each cluster
        core_set_indices = self._select_core_points(n_clusters, points_per_cluster)
        core_set = self.data.loc[core_set_indices]

        # Sort core set by the full 'Time' column, including second precision
        core_set_sorted = core_set.sort_values(by='Time')

        # Export core set sorted by Time
        core_set_sorted.to_csv('core_set_export_sorted.csv', index=False)

        # Visualize clusters
        self.visualization.plot_clusters(X_pca, self.data['Cluster'])
        centroids = kmeans.cluster_centers_
        self.visualization.plot_centroids(centroids)

        logging.info("Core set exported with multiple nodes, sorted by Time (including seconds).")

        return core_set_sorted


    def _select_core_points(self, n_clusters, points_per_cluster):
        """Select representative points per cluster."""
        core_set_indices = []
        for cluster in range(n_clusters):
            cluster_points = self.data[self.data['Cluster'] == cluster]
            if len(cluster_points) >= points_per_cluster:
                selected_points = cluster_points.sample(n=points_per_cluster, random_state=42).index
            else:
                selected_points = cluster_points.index
            core_set_indices.extend(selected_points)
        return core_set_indices

    def grid_search_optimize(self, X_pca, y, max_clusters_range, points_per_cluster_range, time_limit=60):
        """Perform grid search to find the best cluster settings."""
        logging.info("Starting grid search for cluster optimization")
        results = Parallel(n_jobs=-1)(
            delayed(self._evaluate_cluster_params)(X_pca, y, k, p, time_limit)
            for k in max_clusters_range for p in points_per_cluster_range
        )
        best_result = max(results, key=lambda x: x[2])  # Sort by accuracy
        logging.info(f"Best params: max_clusters={best_result[0]}, points_per_cluster={best_result[1]}, Accuracy={best_result[2]:.4f}")
        return best_result[:2], results

    def _evaluate_cluster_params(self, X_pca, y, max_clusters, points_per_cluster, time_limit):
        """Evaluate a specific combination of max_clusters and points_per_cluster."""
        start_time = time.time()
        optimal_k = self.optimize_cluster_selection(X_pca, max_clusters=max_clusters)
        core_set = self.refine_cluster_selection(X_pca, n_clusters=optimal_k, points_per_cluster=points_per_cluster)
        X_core_pca = X_pca[core_set.index]
        y_core = core_set['Value']

        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_core_pca, y_core, test_size=0.2, random_state=42)
        rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
        rf_model.fit(X_train_split, y_train_split)
        y_pred_core = rf_model.predict(X_val_split)
        accuracy = accuracy_score(y_val_split, y_pred_core)

        runtime = time.time() - start_time
        if runtime > time_limit:
            logging.warning(f"Configuration with max_clusters={max_clusters}, points_per_cluster={points_per_cluster} exceeded time limit.")
        return max_clusters, points_per_cluster, accuracy, runtime