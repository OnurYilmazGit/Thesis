import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance
from scipy.spatial.distance import cdist
from data_loader import DataLoader
from benchmarking2 import Benchmarking
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score
import gc
import sys
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import MiniBatchKMeans

gc.enable()

import sys
import os

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


# Suppress warnings
warnings.filterwarnings("ignore")


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
    import os
    import pickle
    import hashlib

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
    import time
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
        return data_final

    # If no cached data, execute step
    print(f"Length of Nodes: {len(nodes)}")
    data_loader = DataLoader(responses_path, sensors_path, nodes)
    # Load responses with adjustments to read 'None' as a string
    responses = data_loader.load_responses(na_values=[], keep_default_na=False)
    sensor_data = data_loader.load_sensors()

    # Merge data on 'Time' and 'Node'
    data = pd.merge(sensor_data, responses, on=['Time', 'Node'])

    # Map application labels to numerical values, including 'None' as 6
    label_mapping = {
        'Kripke': 0,
        'AMG': 1,
        'PENNANT': 2,
        'linpack': 3,
        'LAMMPS': 4,
        'Quicksilver': 5,
        'None': 6,
    }

    # Map labels
    data['Value'] = data['Value'].astype(str)  # Ensure labels are strings
    data['Value'] = data['Value'].map(label_mapping)

    # Get labels per 'Time'
    labels = data.groupby('Time')['Value'].first().reset_index()

    # Drop 'Value' from data
    features = data.drop(columns=['Value'])

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

    end_time = time.time()
    print(f"Step 1 completed in {end_time - start_time:.2f} seconds.")

    return data_final

def step2_time_based_splitting(data_final, split_ratio=0.8):
    import time
    import hashlib
    start_time = time.time()
    print("\n=== Step 2: Time-Based Splitting ===")

    # Define parameters
    # We can create a hash of data_final
    data_hash = hashlib.md5(pd.util.hash_pandas_object(data_final, index=True).values).hexdigest()
    params = {
        'data_hash': data_hash,
        'split_ratio': split_ratio
    }

    # Try to load cached data
    cache_result = cache_step('step2', params, load_only=True)
    if cache_result is not None:
        data_train, data_test = cache_result
        return data_train, data_test

    # If no cached data, execute step
    # Sort data by time
    data_final = data_final.sort_values('Time').reset_index(drop=True)

    # Determine split index
    split_index = int(len(data_final) * split_ratio)

    # Split data
    data_train = data_final.iloc[:split_index].reset_index(drop=True)
    data_test = data_final.iloc[split_index:].reset_index(drop=True)
    print("Training Data Shape:", data_train.shape)
    print("Test Data Shape:", data_test.shape)

    # Save to cache
    cache_step('step2', params, output_data=(data_train, data_test))

    end_time = time.time()
    print(f"Step 2 completed in {end_time - start_time:.2f} seconds.")

    return data_train, data_test

def step3_data_preparation_and_normalization(data_train, data_test, features_to_exclude=['Time', 'Value']):
    import time
    import hashlib
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
        return X_train_full_scaled, X_test_full_scaled, y_train_full, y_test_full, feature_columns

    # Exclude 'Time' from features
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

    end_time = time.time()
    print(f"Step 3 completed in {end_time - start_time:.2f} seconds.")

    return X_train_full_scaled, X_test_full_scaled, y_train_full, y_test_full, feature_columns

def step4_variance_threshold_feature_selection(X_train_full_scaled, X_test_full_scaled, feature_columns, variance_threshold=0.1):
    import time
    import hashlib
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

    end_time = time.time()
    print(f"Step 4 completed in {end_time - start_time:.2f} seconds.")

    return X_train_full_var, X_test_full_var, selected_variance_feature_names

def main():
    # Initialize variables to store execution times
    execution_times = {}

    # Define class names, including 'None' as the 7th label
    class_names = ['Kripke', 'AMG', 'PENNANT', 'linpack', 'LAMMPS', 'Quicksilver', 'None']

    # Parameters for steps
    responses_path = '../responses'
    sensors_path = '../sensors'
    nodes = [f'node{i}' for i in range(4)]  # Use all nodes for a comprehensive analysis

    # Get the node count from the length of the nodes list
    node_count = len(nodes)

    # Add the dynamic output capture
    tee = TeeOutput(node_count)
    sys.stdout = tee  # Redirect all output

    try:
        # Your original code starts here
        print(f"Running process for node count: {node_count}")
        
        # ================================
        # Step 1: Data Loading and Merging
        # ================================
        data_final = step1_data_loading_and_merging(responses_path, sensors_path, nodes)

        # ================================
        # Step 2: Time-Based Splitting
        # ================================
        data_train, data_test = step2_time_based_splitting(data_final, split_ratio=0.8)

        # ================================
        # Step 3: Data Preparation and Normalization
        # ================================
        X_train_full_scaled, X_test_full_scaled, y_train_full, y_test_full, feature_columns = step3_data_preparation_and_normalization(
            data_train, data_test, features_to_exclude=['Time', 'Value']
        )

        # ================================
        # Step 4: Variance Threshold Feature Selection
        # ================================
        X_train_full_var, X_test_full_var, selected_variance_feature_names = step4_variance_threshold_feature_selection(
            X_train_full_scaled, X_test_full_scaled, feature_columns, variance_threshold=0.1
        )

        # Free up memory
        del X_train_full_scaled, X_test_full_scaled
        gc.collect()

        # ================================
        # KMeans Clustering for Core Dataset Selection
        # ================================
        print("\n=== KMeans Clustering for Core Dataset Selection ===")
        start_time = time.time()

        # Decide on the number of clusters
        num_clusters = 500  # Adjust this number as needed
        print(f"Number of clusters: {num_clusters}")

        # Apply KMeans clustering
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=1024, random_state=42)
        kmeans.fit(X_train_full_var)

        # Find the closest data point in each cluster to the cluster centroid
        cluster_centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        # Compute distances from each point to its cluster center
        distances = cdist(cluster_centers, X_train_full_var, 'euclidean')

        # For each cluster, select the point closest to the centroid
        closest_indices = []
        for i in range(num_clusters):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) == 0:
                continue
            cluster_distances = distances[i, cluster_indices]
            closest_index = cluster_indices[np.argmin(cluster_distances)]
            closest_indices.append(closest_index)

        # Create the Core Dataset
        X_core = X_train_full_var[closest_indices]
        y_core = y_train_full.iloc[closest_indices].reset_index(drop=True)

        # Print the Class Distribution in the Core Set
        class_distribution_core = y_core.value_counts(normalize=True)
        print("\nClass distribution in core set (proportion):")
        print(class_distribution_core)

        # Also print counts
        print("\nClass counts in core set:")
        print(y_core.value_counts())

        # Check Core Set Size and Compression Ratio
        core_set_size = X_core.nbytes / 1024  # in KB
        reduction_factor = len(X_train_full_var) / len(X_core)

        end_time = time.time()
        execution_times['KMeans Clustering'] = end_time - start_time
        print(f"KMeans clustering completed in {execution_times['KMeans Clustering']:.2f} seconds.")
        print(f"Core set size: {X_core.shape[0]} samples ({core_set_size:.2f} KB), Reduction factor: {reduction_factor:.2f}")
        print(f"Full set size: {X_train_full_var.nbytes / 1024:.2f} KB")

        # ============================
        # Step 6: Training Models on Core Set Data
        # ============================
        start_time = time.time()
        print("\n=== Step 6: Training Models on Core Set Data ===")

        # Initialize models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=150,
                n_jobs=-1,
                max_depth=20,
                random_state=42,
                class_weight='balanced_subsample',
                bootstrap=True
            )
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
            print(f"\nClassification Report ({model_name} on Test Data):")
            print(classification_report(y_test_full, y_pred, target_names=class_names, zero_division=0))

            # Confusion Matrix
            cm = confusion_matrix(y_test_full, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.show()

        # Record execution time
        end_time = time.time()
        execution_times['Training on Core Set'] = end_time - start_time
        print(f"Models trained and evaluated on core set in {execution_times['Training on Core Set']:.2f} seconds.")

        # Select the Random Forest model trained on the core set for further steps
        rf_core = model_performance['Random Forest']['Model']
        y_pred_core = model_performance['Random Forest']['Predictions']

        # ============================
        # Step 6.5: Further Compression Using Stratified Sampling
        # ============================
        print("\n=== Step 6.5: Further Compression Using Stratified Sampling ===")
        start_time = time.time()

        # Desired compression ratio
        desired_compression_ratio = 2  # Adjust as needed
        sampling_fraction = 1 / desired_compression_ratio  # e.g., 1/100 = 0.01

        # Calculate minimum sampling fraction based on number of classes
        n_classes = len(class_names)  # 7
        min_sampling_fraction = n_classes / len(X_core)  # 7/500 = 0.014

        if sampling_fraction < min_sampling_fraction:
            sampling_fraction = min_sampling_fraction
            print(f"Adjusted sampling fraction to {sampling_fraction:.4f} to ensure at least one sample per class.")
        else:
            print(f"Sampling fraction for desired compression ratio: {sampling_fraction:.4f}")

        # Perform stratified sampling on the core set
        try:
            X_core_sampled, _, y_core_sampled, _ = train_test_split(
                X_core, y_core, 
                train_size=sampling_fraction, 
                stratify=y_core, 
                random_state=42
            )
        except ValueError as e:
            print(f"Error during stratified sampling: {e}")
            print("Adjusting sampling_fraction to meet the minimum class requirement.")
            sampling_fraction = min_sampling_fraction
            X_core_sampled, _, y_core_sampled, _ = train_test_split(
                X_core, y_core, 
                train_size=sampling_fraction, 
                stratify=y_core, 
                random_state=42
            )

        # Verify the size of the compressed core set
        print(f"Compressed core set size: {X_core_sampled.shape[0]} samples")

        # Check class distribution in compressed core set
        print("Class counts in compressed core set:")
        print(y_core_sampled.value_counts())

        # Update compression ratio
        reduction_factor_sampled = len(X_train_full_var) / X_core_sampled.shape[0]
        print(f"Compression Ratio after Stratified Sampling: {reduction_factor_sampled:.2f}")

        # Record execution time for this step
        end_time = time.time()
        execution_times['Stratified Sampling Compression'] = end_time - start_time
        print(f"Stratified sampling compression completed in {execution_times['Stratified Sampling Compression']:.2f} seconds.")

        # ============================
        # Step 6.6: Training Random Forest on Stratified Compressed Core Set
        # ============================
        print("\n=== Step 6.6: Training Random Forest on Stratified Compressed Core Set ===")
        start_time_rf = time.time()

        rf_core_sampled = RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1,
            class_weight='balanced_subsample',
            bootstrap=True,
            random_state=42
        )
        print("Training Random Forest on stratified compressed core set...")
        rf_core_sampled.fit(X_core_sampled, y_core_sampled)

        # Evaluate the model on the test data
        y_pred_core_sampled = rf_core_sampled.predict(X_test_full_var)

        # Adjust target_names based on actual labels present in y_test_full
        labels = np.unique(y_test_full)
        target_names_adjusted = [class_names[int(i)] for i in labels]

        print("\nClassification Report (Stratified Compressed Core Set Model on Test Data):")
        print(classification_report(y_test_full, y_pred_core_sampled, labels=labels, target_names=target_names_adjusted, zero_division=0))

        # Record execution time
        end_time_rf = time.time()
        execution_times['Training on Stratified Compressed Core Set'] = end_time_rf - start_time_rf
        print(f"Random Forest trained and evaluated on stratified compressed core set in {execution_times['Training on Stratified Compressed Core Set']:.2f} seconds.")

        # ============================
        # Step 7: Comparison of Models
        # ============================
        print("\n=== Step 7: Comparison of Models ===")

        # Initialize the full Random Forest model
        rf_full = RandomForestClassifier(
            n_estimators=250,
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

        # Summarize results
        print("\nSummary of Results:")
        print(f"Full Model Test Accuracy: {accuracy_full:.4f}")
        for model_name, perf in model_performance.items():
            print(f"{model_name} Test Accuracy: {perf['Accuracy']:.4f}")

        # =========================
        # Step 8: Statistical Comparison and Summary
        # =========================
        print("\n=== Step 8: Statistical Comparison and Summary ===")

        # Feature count and data size
        full_data_feature_count = X_train_full_var.shape[1]
        core_data_feature_count = X_core.shape[1]
        sampled_core_data_feature_count = X_core_sampled.shape[1]

        summary_df = pd.DataFrame({
            'Dataset': ['Full Data', 'Core Set', 'Stratified Sampled Core'],
            'Samples': [X_train_full_var.shape[0], X_core.shape[0], X_core_sampled.shape[0]],
            'Accuracy': [
                rf_full.score(X_test_full_var, y_test_full),
                rf_core.score(X_test_full_var, y_test_full),
                rf_core_sampled.score(X_test_full_var, y_test_full)
            ],
            'Compression Ratio': [
                1,
                round(reduction_factor, 2),
                round(reduction_factor_sampled, 2)
            ],
            'Data Size (KB)': [
                X_train_full_var.nbytes / 1024,
                X_core.nbytes / 1024,
                X_core_sampled.nbytes / 1024
            ],
            'Number of Features': [
                full_data_feature_count,
                core_data_feature_count,
                sampled_core_data_feature_count
            ]
        })

        print("\n=== Summary Table ===")
        print(summary_df)

        # You can include additional visualizations and statistical tests as in your original code.

        # =========================
        # Final Notes
        # =========================
        print("\n=== All Steps Completed ===")
        print("The code has been modified to use KMeans clustering for core dataset selection. You can experiment with different numbers of clusters to optimize accuracy.")

    finally:
        # Restore stdout
        tee.close()
        sys.stdout = tee.terminal 


# Loop through the node counts and call the main function for each
if __name__ == '__main__':
    main()
