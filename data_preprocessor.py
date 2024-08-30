import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from visualization import Visualization
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class DataPreprocessor:
    def __init__(self, data, window_size=20):
        self.data = data
        self.window_size = window_size
        self.visualization = Visualization()

    def feature_engineering(self):
        features = []
        # Convert the Time column to datetime format
        self.data['Time'] = pd.to_datetime(self.data['Time'], unit='ns')

        # Extracting useful time-based features
        self.data['hour'] = self.data['Time'].dt.hour
        self.data['day'] = self.data['Time'].dt.day
        self.data['month'] = self.data['Time'].dt.month

        # Create rolling mean and standard deviation features for the 'Value' column
        features.append(self.data['Value'].rolling(window=self.window_size).mean().rename('Value_mean'))
        features.append(self.data['Value'].rolling(window=self.window_size).std().rename('Value_std'))

        # Difference feature (rate of change)
        features.append(self.data['Value'].diff().rename('Value_diff'))

        # Concatenate all features at once
        features_df = pd.concat(features, axis=1)
        self.data = pd.concat([self.data, features_df], axis=1)
        self.data.fillna(method='bfill', inplace=True)
        self.data.fillna(method='ffill', inplace=True)

        return self.data
    
    def normalize_data(self, feature_columns):
        scaler = StandardScaler()
        X = self.data[feature_columns]
        X_scaled = scaler.fit_transform(X)
        return X_scaled

    def apply_pca(self, X_scaled, n_components=None):
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Visualization: Explained Variance
        self.visualization.plot_pca_variance(pca)
        
        return X_pca, pca

    def optimize_cluster_selection(self, X_pca, max_clusters=30):
        inertia = []
        K = range(2, max_clusters + 1)

        for k in K:
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=100)
            kmeans.fit(X_pca)
            inertia.append(kmeans.inertia_)

        # Visualization: Elbow Method
        self.visualization.plot_elbow_method(K, inertia)
        
        # Choose the elbow point (you may need to manually adjust this based on the plot)
        optimal_k = self.find_elbow_point(K, inertia)
        return optimal_k

    def find_elbow_point(self, K, inertia):
        derivatives = np.diff(inertia)
        second_derivatives = np.diff(derivatives)
        elbow_index = np.argmax(second_derivatives) + 2  # +2 to adjust for the fact that np.diff reduces array size
        return K[elbow_index]

    def refine_cluster_selection(self, X_pca, n_clusters=10, points_per_cluster=2):
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
        self.data['Cluster'] = kmeans.fit_predict(X_pca)
        
        # Visualization: Show clusters on full dataset
        self.visualization.plot_clusters(X_pca, self.data['Cluster'])
        
        # Core set selection: Select multiple representative points per cluster
        core_set_indices = []
        for cluster in range(n_clusters):
            cluster_points = self.data[self.data['Cluster'] == cluster]
            
            # Check if the cluster has enough points to sample
            if len(cluster_points) >= points_per_cluster:
                selected_points = cluster_points.sample(n=points_per_cluster, random_state=42).index
            else:
                # Select all points if the cluster size is smaller than points_per_cluster
                selected_points = cluster_points.index
            
            core_set_indices.extend(selected_points)

        core_set = self.data.loc[core_set_indices]
        
        # Visualization: Visualize core set within the PCA space
        self.visualization.plot_core_set(X_pca[core_set.index], core_set['Cluster'])
        
        return core_set

    def grid_search_optimize(self, X_pca, y, max_clusters_range, points_per_cluster_range, time_limit=60):
        best_accuracy = 0
        best_params = None
        results = []
        
        for max_clusters in max_clusters_range:
            optimal_k = self.optimize_cluster_selection(X_pca, max_clusters=max_clusters)
            
            for points_per_cluster in points_per_cluster_range:
                start_time = time.time()
                
                # Refine the selection: Choose `points_per_cluster` points per cluster
                core_set = self.refine_cluster_selection(X_pca, n_clusters=optimal_k, points_per_cluster=points_per_cluster)
                X_core_pca = X_pca[core_set.index]
                y_core = core_set['Value']

                # Train and validate model
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_core_pca, y_core, test_size=0.2, random_state=42)
                rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
                rf_model.fit(X_train_split, y_train_split)
                y_pred_core = rf_model.predict(X_val_split)
                accuracy = accuracy_score(y_val_split, y_pred_core)
                
                # Calculate runtime
                runtime = time.time() - start_time
                
                # Record the results
                results.append((max_clusters, points_per_cluster, accuracy, runtime))
                print(f"max_clusters: {max_clusters}, points_per_cluster: {points_per_cluster}, Accuracy: {accuracy}, Runtime: {runtime:.2f} seconds")
                
                # Update best parameters if accuracy is higher and runtime is within acceptable limits
                if accuracy > best_accuracy and runtime <= time_limit:
                    best_accuracy = accuracy
                    best_params = (max_clusters, points_per_cluster)

                # Early stopping if accuracy improvements are minimal
                if len(results) > 1 and accuracy - results[-2][2] < 0.01:
                    break
                    
        return best_params, results
