# data_preprocessor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import logging
from joblib import Parallel, delayed

class DataPreprocessor:
    def __init__(self, data, window_size=10):
        self.data = data
        self.window_size = window_size
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

        # Difference between consecutive values to capture changes
        for lag in range(1, 10):  # Create diff features for 1 to 10 seconds
            self.data[f'diff_{lag}'] = self.data['Value'].diff(lag)
            
        # Forward fill and backward fill
        self.data.fillna(method='ffill', inplace=True)
        self.data.fillna(method='bfill', inplace=True)

        # Log and store data snapshot if needed
        if self.data.empty:
            self.data.to_csv("empty_data_snapshot.csv", index=False)
            raise ValueError("Data is empty after feature engineering. Check data processing steps.")

        logging.info("Feature engineering completed")
        return self.data

    def normalize_data(self, feature_columns):
        """Scale the data using StandardScaler."""
        logging.info("Normalizing data")
        X = self.data[feature_columns].to_numpy(dtype=np.float32)
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled

    def compute_average_path_length(self, rf, X):
        """
        Compute average path length for each sample in Random Forest.

        Args:
            rf (RandomForestClassifier): Trained Random Forest model.
            X (array-like): Feature matrix.

        Returns:
            avg_path_lengths (array): Average path lengths for each sample.
        """
        n_samples = X.shape[0]
        n_estimators = len(rf.estimators_)

        def compute_path_length_for_tree(tree):
            decision_paths = tree.decision_path(X)
            return decision_paths.sum(axis=1).A1  # sum over nodes in path

        # Use Parallel to compute path lengths in parallel
        path_lengths_list = Parallel(n_jobs=-1)(
            delayed(compute_path_length_for_tree)(tree) for tree in rf.estimators_
        )

        # Sum path lengths over all trees
        path_lengths = np.sum(path_lengths_list, axis=0)

        avg_path_lengths = path_lengths / n_estimators
        return avg_path_lengths

    def select_core_set_by_rf(self, X_scaled, y, compression_ratio=0.3):
        logging.info("Selecting core set based on prediction entropy.")

        # Use a simpler Random Forest for importance computation
        rf_full = RandomForestClassifier(
            n_jobs=-1,
            random_state=42
        )
        rf_full.fit(X_scaled, y)

        # Get predicted probabilities
        proba = rf_full.predict_proba(X_scaled)

        # Compute entropy
        entropy_scores = -np.sum(proba * np.log(proba + 1e-9), axis=1)

        # Normalize entropy scores
        entropy_scores_normalized = entropy_scores / np.sum(entropy_scores)

        # Select top samples based on entropy
        n_core_samples = int(len(X_scaled) * compression_ratio)
        indices_sorted = np.argsort(-entropy_scores_normalized)
        core_indices = indices_sorted[:n_core_samples]
        core_set = self.data.iloc[core_indices]

        logging.info(f"Core set selected with {n_core_samples} samples.")

        return core_set


