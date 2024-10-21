import os
import sys
import time
import pickle
import hashlib
import warnings
import gc
from datetime import datetime
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist

from benchmarking2 import Benchmarking

from scipy.stats import (
    ks_2samp,
    chi2_contingency,
    wasserstein_distance,
    ttest_rel,
)
from scipy.spatial.distance import jensenshannon

# Assuming these modules are correctly implemented and available
from data_loader import DataLoader
from benchmarking2 import Benchmarking

# Enable garbage collection
gc.enable()

# Suppress warnings
warnings.filterwarnings("ignore")

# =============================
# Define label_to_class Globally
# =============================
label_to_class = {
    0: 'None',
    1: 'pagefail',
    2: 'leak',
    3: 'ddot',
    4: 'memeater',
    5: 'dial',
    6: 'cpufreq',
    7: 'copy',
    8: 'ioerr'
}

# For reporting purposes, create a list of class names ordered by label
class_names = [label_to_class[i] for i in sorted(label_to_class.keys())]


class TeeOutput:
    def __init__(self, node_count):
        # Create the folder based on node count
        folder_name = f"node_count_{node_count}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        # Generate a timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"time_trained_{timestamp}.txt"
        
        # Define the full file path
        self.file_path = os.path.join(folder_name, filename)
        
        # Open the file in write mode
        self.file = open(self.file_path, 'w')
        
        # Save the original stdout so you can restore it later
        self.terminal = sys.stdout

    def write(self, message):
        # Write both to the terminal and the file
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        # Flush both terminal and file
        self.terminal.flush()
        self.file.flush()

    def close(self):
        # Close the file when done
        self.file.close()


def cache_step(step_name, params, output_data=None, load_only=False):
    """
    Utility function to handle caching of steps.
    - step_name: Name of the step, used to identify cache files.
    - params: Dictionary of parameters used in the step.
    - output_data: Data to be saved (if any).
    - load_only: If True, only attempts to load data, does not save.
    Returns:
    - output_data if loading or after saving.
    """
    cache_dir = 'cache'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Create a unique filename based on step name and parameters
    params_str = str(sorted(params.items()))
    params_hash = hashlib.md5(params_str.encode()).hexdigest()
    cache_filename = os.path.join(cache_dir, f"{step_name}_{params_hash}.pkl")
    params_filename = os.path.join(cache_dir, f"{step_name}_{params_hash}_params.pkl")

    if os.path.exists(cache_filename):
        # Load cached data
        with open(cache_filename, 'rb') as f:
            output_data = pickle.load(f)
        print(f"Loaded cached data for {step_name} with matching parameters.")
        return output_data
    else:
        if load_only:
            print(f"No cached data found for {step_name} with these parameters.")
            return None
        else:
            # Save output_data and params
            with open(cache_filename, 'wb') as f:
                pickle.dump(output_data, f)
            with open(params_filename, 'wb') as f:
                pickle.dump(params, f)
            print(f"Saved data for {step_name} with parameters to cache.")
            return output_data


def step1_data_loading_and_merging(responses_path, sensors_path, nodes):
    start_time = time.time()
    print("\n=== Step 1: Data Loading and Merging ===")

    # Define parameters
    params = {
        'responses_path': responses_path,
        'sensors_path': sensors_path,
        'nodes': nodes
    }

    # Try to load cached data
    data_final = cache_step('step1', params, load_only=True)
    if data_final is not None:
        end_time = time.time()
        print(f"Step 1 completed in {end_time - start_time:.2f} seconds.")
        return data_final

    # If no cached data, execute step
    print(f"Length of Nodes: {len(nodes)}")
    data_loader = DataLoader(responses_path, sensors_path, nodes)
    
    # Load responses and sensors data
    responses = data_loader.load_responses(na_values=[], keep_default_na=False)
    sensor_data = data_loader.load_sensors()

    # Merge data on 'Time' and 'Node'
    data = pd.merge(sensor_data, responses, on=['Time', 'Node'])

    # Map application labels to numerical values, including 'None'
    label_mapping = {
        'None': 0,
        'pagefail': 1,
        'leak': 2,
        'ddot': 3,
        'memeater': 4,
        'dial': 5,
        'cpufreq': 6,
        'copy': 7,
        'ioerr': 8,
    }
    data['Value'] = data['Value'].astype(str)
    data['Value'] = data['Value'].map(label_mapping)

    # Get labels per 'Time'
    labels = data.groupby('Time')['Value'].first().reset_index()

    # Drop 'Value' from data
    features = data.drop(columns=['Value'])

    # **Check for duplicates before pivoting**
    duplicates = features[features.duplicated(subset=['Time', 'Node'], keep=False)]
    if not duplicates.empty:
        print("Duplicates found and handled.")
        # Handle duplicates by aggregating (mean)
        features = features.groupby(['Time', 'Node']).mean().reset_index()

    # Pivot the data to have one row per 'Time' with node metrics as separate columns
    data_pivot = features.pivot(index='Time', columns='Node')

    # Flatten the multi-level columns
    data_pivot.columns = ['{}_{}'.format(node, feature) for feature, node in data_pivot.columns]

    # Reset index
    data_pivot = data_pivot.reset_index()

    # Merge with labels
    data_final = pd.merge(data_pivot, labels, on='Time')

    # Save data to cache
    data_final = cache_step('step1', params, output_data=data_final)

    # Delete intermediate large objects
    del data_loader, responses, sensor_data, data, features, duplicates, data_pivot, labels
    gc.collect()

    end_time = time.time()
    print(f"Step 1 completed in {end_time - start_time:.2f} seconds.")

    return data_final


def step2_stratified_splitting(data_final, test_size=0.2, random_state=42):
    start_time = time.time()
    print("\n=== Step 2: Stratified Splitting ===")

    # Define parameters
    data_hash = hashlib.md5(pd.util.hash_pandas_object(data_final, index=True).values).hexdigest()
    params = {
        'data_hash': data_hash,
        'test_size': test_size,
        'random_state': random_state
    }

    # Try to load cached data
    cache_result = cache_step('step2_stratified', params, load_only=True)
    if cache_result is not None:
        data_train, data_test = cache_result
        end_time = time.time()
        print(f"Step 2 completed in {end_time - start_time:.2f} seconds.")
        return data_train, data_test

    # Split data
    data_train, data_test = train_test_split(
        data_final, test_size=test_size, stratify=data_final['Value'], random_state=random_state
    )
    data_train = data_train.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)
    print("Training Data Shape:", data_train.shape)
    print("Test Data Shape:", data_test.shape)

    # Save to cache
    cache_step('step2_stratified', params, output_data=(data_train, data_test))

    # Delete data_final as it's no longer needed
    del data_final
    gc.collect()

    end_time = time.time()
    print(f"Step 2 completed in {end_time - start_time:.2f} seconds.")

    return data_train, data_test


def step3_data_preparation_and_normalization(data_train, data_test, features_to_exclude=['Time', 'Value']):
    start_time = time.time()
    print("\n=== Step 3: Data Preparation and Normalization ===")

    # Define parameters
    # Hash of data_train and data_test, features_to_exclude
    data_train_hash = hashlib.md5(pd.util.hash_pandas_object(data_train, index=True).values).hexdigest()
    data_test_hash = hashlib.md5(pd.util.hash_pandas_object(data_test, index=True).values).hexdigest()
    params = {
        'data_train_hash': data_train_hash,
        'data_test_hash': data_test_hash,
        'features_to_exclude': features_to_exclude
    }

    # Try to load cached data
    cache_result = cache_step('step3', params, load_only=True)
    if cache_result is not None:
        X_train_full_scaled, X_test_full_scaled, y_train_full, y_test_full, feature_columns = cache_result
        end_time = time.time()
        print(f"Step 3 completed in {end_time - start_time:.2f} seconds.")
        return X_train_full_scaled, X_test_full_scaled, y_train_full, y_test_full, feature_columns

    # Exclude 'Time' and 'Value' from features
    feature_columns = [col for col in data_train.columns if col not in features_to_exclude]

    # Separate features and labels
    X_train_full = data_train[feature_columns]
    y_train_full = data_train['Value']

    X_test_full = data_test[feature_columns]
    y_test_full = data_test['Value']

    # Normalize data
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_test_full_scaled = scaler.transform(X_test_full)

    # Save to cache
    cache_step('step3', params, output_data=(X_train_full_scaled, X_test_full_scaled, y_train_full, y_test_full, feature_columns))

    # Delete original data as it's no longer needed
    del data_train, data_test, X_train_full, X_test_full
    gc.collect()

    end_time = time.time()
    print(f"Step 3 completed in {end_time - start_time:.2f} seconds.")

    return X_train_full_scaled, X_test_full_scaled, y_train_full, y_test_full, feature_columns


def step4_variance_threshold_feature_selection(X_train_full_scaled, X_test_full_scaled, feature_columns, variance_threshold=0.0001):
    start_time = time.time()
    print("\n=== Step 4: Variance Threshold Feature Selection ===")

    # Define parameters
    # Hash of X_train_full_scaled
    X_train_full_scaled_hash = hashlib.md5(np.ascontiguousarray(X_train_full_scaled)).hexdigest()
    params = {
        'X_train_full_scaled_hash': X_train_full_scaled_hash,
        'variance_threshold': variance_threshold
    }

    # Try to load cached data
    cache_result = cache_step('step4', params, load_only=True)
    if cache_result is not None:
        X_train_full_var, X_test_full_var, selected_variance_feature_names = cache_result
        end_time = time.time()
        print(f"Step 4 completed in {end_time - start_time:.2f} seconds.")
        return X_train_full_var, X_test_full_var, selected_variance_feature_names

    # Initialize the variance threshold selector
    selector = VarianceThreshold(threshold=variance_threshold)

    # Fit and transform the training data
    X_train_full_var = selector.fit_transform(X_train_full_scaled)
    selected_variance_features = selector.get_support(indices=True)
    selected_variance_feature_names = [feature_columns[i] for i in selected_variance_features]
    # Apply the same transformation to test data
    X_test_full_var = selector.transform(X_test_full_scaled)

    # Calculate the number of features before and after
    print(f"Number of features before Variance Threshold: {X_train_full_scaled.shape[1]}")
    print(f"Number of features after Variance Threshold: {X_train_full_var.shape[1]}")

    # Save to cache
    cache_step('step4', params, output_data=(X_train_full_var, X_test_full_var, selected_variance_feature_names))

    # Delete scaled data as it's no longer needed
    del X_train_full_scaled, X_test_full_scaled, selector
    gc.collect()

    end_time = time.time()
    print(f"Step 4 completed in {end_time - start_time:.2f} seconds.")

    return X_train_full_var, X_test_full_var, selected_variance_feature_names


def step5_kmeans_clustering(X_train_full_var, y_train_full, num_clusters=200, batch_size=1024, random_state=42):
    """
    Step 5: KMeans Clustering for Core Dataset Selection
    Selects a representative core set using KMeans clustering.
    """
    start_time = time.time()
    print("\n=== Step 5: KMeans Clustering for Core Dataset Selection ===")

    # Define parameters for caching
    params = {
        'num_clusters': num_clusters,
        'batch_size': batch_size,
        'random_state': random_state
    }

    # Try to load cached core set
    cache_result = cache_step('step5_kmeans_clustering', params, load_only=True)
    if cache_result is not None:
        X_core, y_core = cache_result
        print("Loaded cached core set from KMeans clustering.")
        end_time = time.time()
        print(f"Step 5 completed in {end_time - start_time:.2f} seconds.")
        return X_core, y_core

    # Apply KMeans clustering
    print(f"Number of clusters: {num_clusters}")
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=batch_size, random_state=random_state)
    kmeans.fit(X_train_full_var)

    # Find the closest data point in each cluster to the cluster centroid
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Compute distances from each point to its cluster center
    # Efficient computation using NumPy broadcasting
    closest_indices = []
    for i in range(num_clusters):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) == 0:
            continue
        cluster_points = X_train_full_var[cluster_indices]
        distances = np.linalg.norm(cluster_points - cluster_centers[i], axis=1)
        closest_index = cluster_indices[np.argmin(distances)]
        closest_indices.append(closest_index)

    # Create the Core Dataset
    X_core = X_train_full_var[closest_indices]
    y_core = y_train_full.iloc[closest_indices].reset_index(drop=True)

    # Print the Class Distribution in the Core Set
    class_distribution_core = y_core.value_counts(normalize=True).sort_index()
    print("\nClass distribution in core set (proportion):")
    for label, proportion in class_distribution_core.items():
        print(f"Class {label} ({label_to_class[label]}): {proportion:.6f}")

    # Also print counts
    print("\nClass counts in core set:")
    print(y_core.value_counts().sort_index())

    # Check Core Set Size and Compression Ratio
    core_set_size = X_core.nbytes / 1024  # in KB
    reduction_factor = len(X_train_full_var) / len(X_core)
    print(f"Core set size: {X_core.shape[0]} samples ({core_set_size:.2f} KB), Reduction factor: {reduction_factor:.2f}")
    print(f"Full set size: {X_train_full_var.nbytes / 1024:.2f} KB")

    # Save the core set to cache
    cache_step('step5_kmeans_clustering', params, output_data=(X_core, y_core))

    # Record execution time
    end_time = time.time()
    execution_times['KMeans Clustering'] = end_time - start_time
    print(f"KMeans clustering completed in {execution_times['KMeans Clustering']:.2f} seconds.")

    return X_core, y_core


def step6_training_models_on_core_set(X_core, y_core, X_test_full_var, y_test_full, class_names):
    """
    Step 6: Training Models on Core Set Data
    Trains specified models on the core set and evaluates them on the test set.
    """
    start_time = time.time()
    print("\n=== Step 6: Training Models on Core Set Data ===")

    # Initialize models with class_weight='balanced_subsample' to handle class imbalance
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1,
            class_weight='balanced_subsample',
            bootstrap=True,
            random_state=42
        )
        # You can add more models here if needed
    }

    model_performance = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name} on core set...")
        model.fit(X_core, y_core)
        y_pred = model.predict(X_test_full_var)
        accuracy = accuracy_score(y_test_full, y_pred)
        f1 = f1_score(y_test_full, y_pred, average='weighted')
        model_performance[model_name] = {
            'Model': model,
            'Accuracy': accuracy,
            'F1-Score': f1,
            'Predictions': y_pred
        }
        print(f"{model_name} Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")

        # Classification Report
        unique_labels = np.unique(y_test_full)

        # Verify that all labels are within the expected range
        expected_labels = set(label_to_class.keys())
        actual_labels = set(unique_labels)

        if not actual_labels.issubset(expected_labels):
            unexpected_labels = actual_labels - expected_labels
            raise ValueError(f"Unexpected labels found in test set: {unexpected_labels}")

        # Adjust the target names using the dictionary
        adjusted_target_names = [label_to_class[int(i)] for i in unique_labels]

        # Print the classification report with adjusted target names
        print(classification_report(y_test_full, y_pred, labels=unique_labels, target_names=adjusted_target_names, zero_division=0))

        # Confusion Matrix
        cm = confusion_matrix(y_test_full, y_pred, labels=unique_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=adjusted_target_names, yticklabels=adjusted_target_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
        plt.close()  # Free up memory

    # Record execution time
    end_time = time.time()
    execution_times['Training on Core Set'] = end_time - start_time
    print(f"Models trained and evaluated on core set in {execution_times['Training on Core Set']:.2f} seconds.")

    # Select the Random Forest model trained on the core set for further steps
    rf_core = model_performance['Random Forest']['Model']
    y_pred_core = model_performance['Random Forest']['Predictions']

    return rf_core, y_pred_core, model_performance


def step6_pca_compression(X_core, X_test_full_var, desired_variance=0.95):
    """
    Step 6.5: PCA-Based Compression
    Applies PCA to the core dataset to reduce dimensionality while retaining the desired variance.
    """
    start_time = time.time()
    print("\n=== Step 6.5: PCA-Based Compression ===")

    # Define PCA parameters
    print(f"Applying PCA to retain {desired_variance * 100}% of variance")

    # Define parameters for caching
    params = {
        'desired_variance': desired_variance,
        'num_features': X_core.shape[1]
    }

    # Try to load cached PCA
    cache_result = cache_step('step6_pca_compression', params, load_only=True)
    if cache_result is not None:
        X_core_pca, X_test_pca, pca = cache_result
        print("Loaded cached PCA model.")
        end_time = time.time()
        print(f"Step 6.5 completed in {end_time - start_time:.2f} seconds.")
        return X_core_pca, X_test_pca, pca

    # Initialize PCA
    pca = PCA(n_components=desired_variance, random_state=42)

    # Fit PCA on the core set
    pca.fit(X_core)

    # Transform core set and test set
    X_core_pca = pca.transform(X_core)
    X_test_pca = pca.transform(X_test_full_var)

    print(f"PCA reduced the feature dimensions from {X_core.shape[1]} to {pca.n_components_}")

    # Calculate the reduction factor
    reduction_factor_pca = X_core.shape[1] / pca.n_components_

    # Log the compression ratio
    print(f"Compression Ratio after PCA: {reduction_factor_pca:.2f}")

    # Save to cache
    cache_step('step6_pca_compression', params, output_data=(X_core_pca, X_test_pca, pca))

    # Delete original large matrices
    del X_core, X_test_full_var, pca
    gc.collect()

    end_time = time.time()
    print(f"Step 6.5 completed in {end_time - start_time:.2f} seconds.")

    return X_core_pca, X_test_pca, pca



def main():
    # Initialize variables to store execution times
    global execution_times
    execution_times = {}

    # Parameters for steps
    responses_path = '../responses'
    sensors_path = '../sensors'
    nodes = [f'node{i}' for i in range(1)]  # Adjust the range as per your data

    # Get the node count from the length of the nodes list
    node_count = len(nodes)

    # Add the dynamic output capture
    tee = TeeOutput(node_count)
    sys.stdout = tee  # Redirect all output

    try:
        print(f"Running process for node count: {node_count}")

        # ================================
        # Step 1: Data Loading and Merging
        # ================================
        data_final = step1_data_loading_and_merging(responses_path, sensors_path, nodes)
        gc.collect()

        # ================================
        # Step 2: Stratified Splitting
        # ================================
        data_train, data_test = step2_stratified_splitting(data_final, test_size=0.2, random_state=42)
        gc.collect()

        # ================================
        # Data Integrity Check
        # ================================
        print("\n=== Data Integrity Check ===")
        unique_train_labels = sorted(data_train['Value'].unique())
        unique_test_labels = sorted(data_test['Value'].unique())
        print(f"Unique labels in training data: {unique_train_labels}")
        print(f"Unique labels in testing data: {unique_test_labels}")

        expected_labels = set(label_to_class.keys())
        actual_train_labels = set(unique_train_labels)
        actual_test_labels = set(unique_test_labels)

        if not actual_train_labels == expected_labels:
            missing_train = expected_labels - actual_train_labels
            print(f"Warning: Training data is missing labels: {missing_train}")

        if not actual_test_labels == expected_labels:
            missing_test = expected_labels - actual_test_labels
            print(f"Warning: Testing data is missing labels: {missing_test}")

        # ================================
        # Step 3: Data Preparation and Normalization
        # ================================
        X_train_full_scaled, X_test_full_scaled, y_train_full, y_test_full, feature_columns = step3_data_preparation_and_normalization(
            data_train, data_test, features_to_exclude=['Time', 'Value']
        )
        gc.collect()

        # ================================
        # Step 4: Variance Threshold Feature Selection
        # ================================
        X_train_full_var, X_test_full_var, selected_variance_feature_names = step4_variance_threshold_feature_selection(
            X_train_full_scaled, X_test_full_scaled, feature_columns, variance_threshold=0.0001
        )
        gc.collect()

        # ================================
        # Step 5: KMeans Clustering for Core Dataset Selection
        # ================================

        desired_core_size = len(X_train_full_var) // 100  # Adjust as needed

        X_core, y_core = step5_kmeans_clustering(
            X_train_full_var, y_train_full, num_clusters=desired_core_size, batch_size=1024, random_state=42
        )
        gc.collect()

        # ============================
        # Step 6: Training Models on Core Set Data
        # ============================
        rf_core, y_pred_core, model_performance = step6_training_models_on_core_set(
            X_core, y_core, X_test_full_var, y_test_full, class_names
        )
        gc.collect()

        # ============================
        # Step 6.5: PCA-Based Compression
        # ============================
        X_core_pca, X_test_pca, pca = step6_pca_compression(
            X_core, X_test_full_var, desired_variance=0.95
        )
        gc.collect()

        # ============================
        # Step 7: Training Models on PCA-Compressed Core Set
        # ============================
        print("\n=== Step 7: Training Models on PCA-Compressed Core Set ===")
        start_time_rf_pca = time.time()

        rf_pca = RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1,
            class_weight='balanced_subsample',
            bootstrap=True,
            random_state=42
        )
        print("Training Random Forest on PCA-compressed core set...")
        rf_pca.fit(X_core_pca, y_core)

        # Evaluate the model on the test data
        y_pred_pca = rf_pca.predict(X_test_pca)

        # Adjust target_names based on actual labels present in y_test_full
        labels = np.unique(y_test_full)
        target_names_adjusted = [label_to_class[int(i)] for i in labels]

        print("\nClassification Report (PCA-Compressed Core Set Model on Test Data):")
        print(classification_report(y_test_full, y_pred_pca, labels=labels, target_names=target_names_adjusted, zero_division=0))

        # Record execution time
        end_time_rf_pca = time.time()
        execution_times['Training on PCA Compressed Core Set'] = end_time_rf_pca - start_time_rf_pca
        print(f"Random Forest trained and evaluated on PCA-compressed core set in {execution_times['Training on PCA Compressed Core Set']:.2f} seconds.")

        # ============================
        # Step 8: Comparison of Models
        # ============================
        print("\n=== Step 8: Comparison of Models ===")

        # Initialize the full Random Forest model
        rf_full = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced_subsample',
            min_samples_leaf=5,
            criterion='entropy',
            bootstrap=True,
        )

        # Train the full Random Forest model on the full training data
        print("\nTraining Full Random Forest Model on Full Training Data...")
        rf_full.fit(X_train_full_var, y_train_full)
        print("Training completed.")

        # Evaluate the full Random Forest model on the test data
        print("Evaluating Full Random Forest Model on Test Data...")
        y_pred_full = rf_full.predict(X_test_full_var)
        accuracy_full = accuracy_score(y_test_full, y_pred_full)
        f1_full = f1_score(y_test_full, y_pred_full, average='weighted')

        print(f"Full Model Accuracy: {accuracy_full:.4f}, F1-Score: {f1_full:.4f}")

        # Confusion Matrix for Full Model
        cm_full = confusion_matrix(y_test_full, y_pred_full)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_full, annot=True, fmt='d', cmap='Greens',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - Full Random Forest Model')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
        plt.close()  # Free up memory

        # Summarize results
        print("\nSummary of Results:")
        print(f"Full Model Test Accuracy: {accuracy_full:.4f}")
        for model_name, perf in model_performance.items():
            print(f"{model_name} Test Accuracy: {perf['Accuracy']:.4f}")
        print(f"PCA Compressed Model Test Accuracy: {accuracy_score(y_test_full, y_pred_pca):.4f}")

        # =========================
        # Step 9: Statistical Comparison and Summary
        # =========================
        print("\n=== Step 8: Statistical Comparison and Summary ===")
        reduction_factor = len(X_train_full_var) / len(X_core)
        reduction_factor_pca = len(X_train_full_var) / len(X_core_pca)

        # Feature count and data size
        full_data_feature_count = X_train_full_var.shape[1]
        core_data_feature_count = X_core.shape[1]
        pca_compressed_feature_count = X_core_pca.shape[1]

        summary_df = pd.DataFrame({
            'Dataset': ['Full Data', 'Core Set', 'PCA Compressed Core'],
            'Samples': [X_train_full_var.shape[0], X_core.shape[0], X_core_pca.shape[0]],
            'Accuracy': [
                rf_full.score(X_test_full_var, y_test_full),
                rf_core.score(X_test_full_var, y_test_full),
                rf_pca.score(X_test_pca, y_test_full)
            ],
            'Compression Ratio': [
                1,
                round(reduction_factor, 2),
                round(reduction_factor_pca, 2)
            ],
            'Data Size (KB)': [
                X_train_full_var.nbytes / 1024,
                X_core.nbytes / 1024,
                X_core_pca.nbytes / 1024
            ],
            'Number of Features': [
                full_data_feature_count,
                core_data_feature_count,
                pca_compressed_feature_count
            ]
        })

        print("\n=== Summary Table ===")
        print(summary_df)

        benchmark = Benchmarking(
            X_full=X_train_full_var,
            y_full=y_train_full,
            X_core=X_core,
            y_core=y_core,
            X_compressed=X_core_pca,
            y_compressed=y_core,
            X_test=X_test_full_var,
            y_test=y_test_full,
            feature_names=selected_variance_feature_names,
            class_names=class_names
        )

        # =========================
        # Step 11: Detailed Feature Similarity Logging
        # =========================
        print("\n=== Step 11: Detailed Feature Similarity Logging ===")
        start_time = time.time()

        # 1. Get Feature Importances from the Full Model
        print("\nCalculating feature importances from the full model...")
        feature_importances = rf_full.feature_importances_

        # Create a DataFrame for feature importances
        feature_importance_df = pd.DataFrame({
            'Feature': selected_variance_feature_names,
            'Importance': feature_importances
        })

        # Sort features by importance
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        # Select top N important features
        top_n = 50  # You can adjust this number
        top_features = feature_importance_df.head(top_n)['Feature'].tolist()
        top_feature_indices = [selected_variance_feature_names.index(f) for f in top_features]

        # 2. Log Similarity Measures for Top Features
        similarity_logs = []
        print(f"\nLogging similarity measures for top {top_n} important features...")
        for idx in top_feature_indices:
            feature_name = selected_variance_feature_names[idx]
            # Kolmogorov-Smirnov Test
            if idx < X_core_pca.shape[1]:
                stat_ks, p_value_ks = ks_2samp(X_train_full_var[:, idx], X_core_pca[:, idx % X_core_pca.shape[1]])
            else:
                stat_ks, p_value_ks = ks_2samp(X_train_full_var[:, idx], np.zeros_like(X_train_full_var[:, idx]))
            # Jensen-Shannon Divergence
            hist_full, bin_edges = np.histogram(X_train_full_var[:, idx], bins=50, density=True)
            if idx < X_core_pca.shape[1]:
                hist_compressed, _ = np.histogram(X_core_pca[:, idx % X_core_pca.shape[1]], bins=bin_edges, density=True)
            else:
                hist_compressed = np.zeros_like(hist_full)
            # Add a small value to avoid zeros
            hist_full += 1e-8
            hist_compressed += 1e-8
            js_distance = jensenshannon(hist_full, hist_compressed)
            # Wasserstein Distance
            if idx < X_core_pca.shape[1]:
                wasserstein_dist = wasserstein_distance(X_train_full_var[:, idx], X_core_pca[:, idx % X_core_pca.shape[1]])
            else:
                wasserstein_dist = wasserstein_distance(X_train_full_var[:, idx], np.zeros_like(X_train_full_var[:, idx]))
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
        similarity_logs_df.to_csv('feature_similarity_logs_pca.csv', index=False)
        print("\nFeature similarity logs saved to 'feature_similarity_logs_pca.csv'.")

        # 3. Display the logs
        print("\nTop Features Similarity Measures:")
        print(similarity_logs_df.head(10))  # Display top 10 for brevity

        # 4. Visualize Distributions of Top Features
        print("\nVisualizing distributions of top important features...")
        for idx in top_feature_indices[:5]:  # Visualize top 5 features
            feature_name = selected_variance_feature_names[idx]
            plt.figure(figsize=(8, 4))
            sns.kdeplot(X_train_full_var[:, idx], label='Full Data', shade=True)
            if idx < X_core_pca.shape[1]:
                sns.kdeplot(X_core_pca[:, idx % X_core_pca.shape[1]], label='PCA Compressed Data', shade=True)
            plt.title(f'Distribution Comparison for Feature: {feature_name}')
            plt.legend()
            plt.show()
            plt.close()  # Free up memory

        # Record execution time
        end_time = time.time()
        execution_times['Feature Similarity Logging'] = end_time - start_time
        print(f"Feature similarity logging completed in {execution_times['Feature Similarity Logging']:.2f} seconds.")

        # =========================
        # Step 12: Memory and Computational Efficiency
        # =========================
        print("\n=== Step 12: Memory and Computational Efficiency ===")
        from sys import getsizeof

        memory_full = getsizeof(X_train_full_var) / 1024 ** 2  # Convert to MB
        memory_compressed = getsizeof(X_core_pca) / 1024 ** 2
        memory_test_compressed = getsizeof(X_test_pca) / 1024 ** 2
        time_full = execution_times.get('KMeans Clustering', 0) + execution_times.get('Training on Core Set', 0)
        time_compressed = execution_times.get('Training on PCA Compressed Core Set', 0)
        time_full_model = execution_times.get('Training on Core Set', 0) + execution_times.get('Training on PCA Compressed Core Set', 0)

        print(f"Memory Usage - Full Dataset: {memory_full:.2f} MB")
        print(f"Memory Usage - PCA Compressed Core Dataset: {memory_compressed:.2f} MB")
        print(f"Memory Usage - PCA Compressed Test Dataset: {memory_test_compressed:.2f} MB")
        print(f"Training Time - Core Model: {time_full:.2f} seconds")
        print(f"Training Time - PCA Compressed Core Model: {time_compressed:.2f} seconds")
        print(f"Total Training Time: {time_full_model:.2f} seconds")

        # =========================
        # Step 13: Benchmarking
        # =========================
        print("\n=== Step 13: Benchmarking ===")
        start_time = time.time()

        # Call all benchmarking functions
        print("\n--- Comparing Model Performance ---")
        benchmark.compare_model_performance(rf_full, rf_core, rf_pca)

        print("\n--- Performing Statistical Similarity Tests ---")
        benchmark.statistical_similarity_tests()

        print("\n--- Comparing Feature Importances ---")
        benchmark.feature_importance_comparison(rf_full, rf_core)

        print("\n--- Visualizing Feature Distributions ---")
        benchmark.visualize_feature_distributions()

        print("\n--- Evaluating Clustering Metrics ---")
        benchmark.clustering_evaluation()

        print("\n--- Performing Cross-Validation Checks ---")
        benchmark.cross_validation_checks(rf_core)

        print("\n--- Checking for Data Leakage ---")
        benchmark.check_data_leakage()

        # Record execution time
        end_time = time.time()
        execution_times['Benchmarking'] = end_time - start_time
        print(f"Benchmarking completed in {execution_times['Benchmarking']:.2f} seconds.")


        # =========================
        # Final Notes
        # =========================
        print("\n=== All Steps Completed ===")
        print("The code has been modified to use PCA-based compression for dimensionality reduction instead of stratified sampling. You can experiment with different PCA variance thresholds to optimize the balance between compression and model performance.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Restore stdout
        tee.close()
        sys.stdout = tee.terminal


# Ensure that only one main call exists
if __name__ == '__main__':
    main()
