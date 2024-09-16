# dataPreprocessor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedShuffleSplit
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

    def select_features(self, X, y, k=50):
        """Select top k features based on ANOVA F-score."""
        logging.info(f"Selecting top {k} features")
        selector = SelectKBest(f_classif, k=k)
        X_new = selector.fit_transform(X, y)
        selected_features = selector.get_support(indices=True)
        return X_new, selected_features

    def select_core_set_by_rf(self, X_scaled, y, data, compression_ratio=0.01):
        """Select core set using stratified sampling to ensure class balance."""
        logging.info("Selecting core set using stratified sampling based on compression ratio.")
        
        # Compute the number of samples per class
        class_counts = y.value_counts()
        min_samples_per_class = 10  # Set a reasonable minimum
        n_core_samples = int(len(X_scaled) * compression_ratio)

        # Adjust n_core_samples if necessary
        n_classes = len(class_counts)
        if n_core_samples < n_classes * min_samples_per_class:
            n_core_samples = n_classes * min_samples_per_class

        # Stratified sampling
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_core_samples, random_state=42)
        for core_indices, _ in sss.split(X_scaled, y):
            core_set = data.iloc[core_indices].copy()
            X_core = X_scaled[core_indices]
            y_core = y.iloc[core_indices].reset_index(drop=True)
            break

        logging.info(f"Core set selected with {len(core_indices)} samples.")

        return core_set, X_core, y_core
