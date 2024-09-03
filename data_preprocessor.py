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
        """Create features such as rolling mean, std, and diff based on a window size."""
        logging.info("Starting feature engineering")
        time_col = pd.to_datetime(self.data['Time'], unit='ns')
        self.data['minute'] = time_col.dt.minute

        def calculate_rolling_mean():
            return self.data['Value'].rolling(window=self.window_size).mean().rename('Value_mean')

        def calculate_rolling_std():
            return self.data['Value'].rolling(window=self.window_size).std().rename('Value_std')

        def calculate_diff():
            return self.data['Value'].diff().rename('Value_diff')

        # Parallelize the feature calculation
        rolling_mean, rolling_std, diff_feature = Parallel(n_jobs=-1)(
            delayed(func)() for func in [calculate_rolling_mean, calculate_rolling_std, calculate_diff]
        )

        # Concatenate features
        features_df = pd.concat([rolling_mean, rolling_std, diff_feature], axis=1)
        self.data = pd.concat([self.data, features_df], axis=1).fillna(method='bfill').fillna(method='ffill')

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
        return X_scaled


    def apply_pca(self, X_scaled, n_components=None):
        """Apply PCA to the scaled data."""
        logging.info("Applying PCA")
        if self.use_incremental_pca:
            pca = IncrementalPCA(n_components=n_components, batch_size=500)
        else:
            pca = PCA(n_components=n_components)

        X_pca = pca.fit_transform(X_scaled)
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

    def refine_cluster_selection(self, X_pca, n_clusters=20, points_per_cluster=20):
        """Refine the cluster selection by choosing representative points."""
        logging.info(f"Refining cluster selection with {n_clusters} clusters")
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
        self.data['Cluster'] = kmeans.fit_predict(X_pca)

        core_set_indices = self._select_core_points(n_clusters, points_per_cluster)
        core_set = self.data.loc[core_set_indices]
        return core_set

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
