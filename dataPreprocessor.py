# dataPreprocessor.py

import pandas as pd
import numpy as np
import logging
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import _tree

class DataPreprocessor:
    def __init__(self, data, window_size=10):
        self.data = data
        self.window_size = window_size

    def feature_engineering(self):
        """Create features such as lagged values and differences based on seconds."""
        logging.info("Starting feature engineering")
        
        # Log the initial shape of the data
        print(f"Initial data shape: {self.data.shape}")

        # Ensure data is sorted by 'Node' and 'Time'
        self.data.sort_values(by=['Node', 'Time'], inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        
        # Lag and diff features per 'Node'
        for lag in range(1, 10):
            self.data[f'lag_{lag}'] = self.data.groupby('Node')['Value'].shift(lag)
            self.data[f'diff_{lag}'] = self.data.groupby('Node')['Value'].diff(lag)     

        # Do not fill missing values to avoid data leakage
        # Alternatively, handle missing values carefully
        # For training data
        if self.data['lag_1'].isnull().any():
            self.data['lag_1'].fillna(method='bfill', inplace=True)
        # For test data, do not use any values from training data
        # Leave NaNs as is or fill with a constant value if appropriate

        if self.data.empty:
            self.data.to_csv("empty_data_snapshot.csv", index=False)
            raise ValueError("Data is empty after feature engineering. Check data processing steps.")

        self.data.to_csv("empty_data_snapshot2.csv", index=False)

        logging.info("Feature engineering completed")
        return self.data


    def select_features(self, X, y, k=300):
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
        class_counts = pd.Series(y).value_counts()
        min_samples_per_class = 10  # Set a reasonable minimum
        n_classes = len(class_counts)
        n_core_samples = int(len(X_scaled) * compression_ratio)

        # Adjust n_core_samples if necessary
        if n_core_samples < n_classes * min_samples_per_class:
            n_core_samples = n_classes * min_samples_per_class

        # Stratified sampling
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_core_samples, random_state=42)
        for core_indices, _ in sss.split(X_scaled, y):
            core_set = data.iloc[core_indices].copy()
            X_core = X_scaled[core_indices]
            y_core = y[core_indices]
            break

        logging.info(f"Core set selected with {len(core_indices)} samples.")

        return core_set, X_core, y_core
        

    def select_features_rfe(self, X, y, n_features_to_select=None, estimator=None, step=1):
        logging.info("Selecting features using RFE")

        if estimator is None:
            estimator = RandomForestClassifier(n_jobs=-1, random_state=42)

        selector = RFE(estimator, n_features_to_select=n_features_to_select, step=step)
        selector.fit(X, y)
        X_new = selector.transform(X)
        selected_features = selector.get_support(indices=True)

        logging.info(f"Selected {len(selected_features)} features using RFE.")

        return X_new, selected_features

    def variance_threshold_feature_selection(X_train, X_test, threshold):
        # Initialize the variance threshold selector
        selector = VarianceThreshold(threshold=threshold)
        
        # Fit and transform the training data
        X_train_selected = selector.fit_transform(X_train)
        
        # Apply the same transformation to the test data
        X_test_selected = selector.transform(X_test)
        
        # Return only the selected features and the transformed data
        return X_train_selected, X_test_selected


    def export_feature_importances(rf_model, feature_columns, filename="feature_importances.csv"):
        # Get feature importances from the model
        importances = rf_model.feature_importances_
        
        # Create a DataFrame for feature importances
        feature_importances = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': importances
        })
        
        # Sort by importance and save to CSV
        feature_importances.sort_values(by='Importance', ascending=False, inplace=True)
        feature_importances.to_csv(filename, index=False)
        print(f"Feature importances exported to {filename}")


    def get_high_entropy_samples(self, rf_model, X_train, y_train, percentile=95):
        """Extract samples that reach high-entropy leaf nodes in the Random Forest."""
        print("Extracting high-entropy samples from Random Forest")

        # Collect impurities from leaf nodes across all trees
        leaf_impurities = []

        for tree in rf_model.estimators_:
            tree_struct = tree.tree_
            # Get leaf node indices
            leaf_indices = np.where(tree_struct.children_left == _tree.TREE_LEAF)[0]
            # Get impurities (entropy) of leaf nodes
            leaf_impurities.extend(tree_struct.impurity[leaf_indices])

        # Determine impurity threshold based on the desired percentile
        impurity_threshold = np.percentile(leaf_impurities, percentile)
        print(f"Impurity threshold (percentile {percentile}%): {impurity_threshold}")

        # Initialize set to hold indices of high-entropy samples
        high_entropy_indices = set()

        for tree_idx, tree in enumerate(rf_model.estimators_):
            tree_struct = tree.tree_
            # Get leaf node indices
            leaf_indices = np.where(tree_struct.children_left == _tree.TREE_LEAF)[0]
            # Get impurities (entropy) of leaf nodes
            leaf_impurities = tree_struct.impurity[leaf_indices]

            # Select high-entropy leaf nodes
            high_entropy_leaf_nodes = leaf_indices[leaf_impurities >= impurity_threshold]
            if len(high_entropy_leaf_nodes) == 0:
                continue

            # Use the decision path to find samples that reach high-entropy leaf nodes
            decision_paths = tree.decision_path(X_train)
            for node_id in high_entropy_leaf_nodes:
                # Get samples that reach this node
                sample_indices = np.where(decision_paths[:, node_id].toarray().ravel() == 1)[0]
                high_entropy_indices.update(sample_indices)

        print(f"Total high-entropy samples selected: {len(high_entropy_indices)}")
        return list(high_entropy_indices)
    

    def calculate_sample_entropy(rf_model, X):
        """Calculate entropy for each sample based on predicted probabilities."""
        # Get predicted class probabilities
        probs = rf_model.predict_proba(X)
        # Calculate entropy
        entropy = -np.sum(probs * np.log(probs + 1e-9), axis=1)  # Add small value to avoid log(0)
        return entropy
