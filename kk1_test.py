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
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from scipy.special import entr
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance, ttest_rel
from scipy.spatial.distance import jensenshannon
from data_loader import DataLoader
from benchmarking2 import Benchmarking
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, f1_score
import gc
import sys
import os
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
    nodes = [f'node{i}' for i in range(16)]  # Use all nodes for a comprehensive analysis

    # Get the node count from the length of the nodes list
    node_count = len(nodes)

    # Add the dynamic output capture
    tee = TeeOutput(node_count)
    sys.stdout = tee  # Redirect all output


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
    # Optimized Entropy-Driven Sampling Strategy with Predictive Uncertainties
    # ================================
    print("\n=== Optimized Entropy-Driven Sampling Strategy ===")
    start_time = time.time()

    # Step 1: Train Random Forest on the full training data with variance-thresholded features
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
    print("Training Random Forest on variance-thresholded full training data...")
    rf_full.fit(X_train_full_var, y_train_full)
    print("Random Forest training completed.")

    # Step 2: Compute Predictive Uncertainties using Entropy
    # Get predicted class probabilities for training data
    probs = rf_full.predict_proba(X_train_full_var)
    print("Predicted class probabilities shape:", probs.shape)

    # Compute entropy for each sample
    predictive_entropies = entr(probs).sum(axis=1)
    print("Predictive entropies shape:", predictive_entropies.shape)

    # Determine thresholds
    high_uncertainty_threshold = np.percentile(predictive_entropies, 99)
    low_uncertainty_threshold = np.percentile(predictive_entropies, 1)

    # Select samples
    high_uncertainty_indices = np.where(predictive_entropies >= high_uncertainty_threshold)[0]
    low_uncertainty_indices = np.where(predictive_entropies <= low_uncertainty_threshold)[0]

    # Combine indices
    selected_indices = np.concatenate([high_uncertainty_indices, low_uncertainty_indices])

    # Step 4: Ensure Balanced Class Distribution
    # Calculate Class Proportions in Full Dataset
    class_distribution_full = y_train_full.value_counts(normalize=True)
    print("\nClass distribution in full dataset (proportion):")
    print(class_distribution_full)

    # Create the Core Dataset
    X_core = X_train_full_var[selected_indices]
    y_core = y_train_full.iloc[selected_indices].reset_index(drop=True)

    # Step 6: Print the Class Distribution in the Core Set
    class_distribution_core = y_core.value_counts(normalize=True)
    print("\nClass distribution in core set (proportion):")
    print(class_distribution_core)

    # Also print counts
    print("\nClass counts in core set:")
    print(y_core.value_counts())

    # Step 7: Check Core Set Size and Compression Ratio
    core_set_size = X_core.nbytes / 1024  # in KB
    reduction_factor = len(X_train_full_var) / len(X_core)

    end_time = time.time()
    execution_times['Optimized Entropy-Driven Sampling'] = end_time - start_time
    print(f"Optimized entropy-driven sampling completed in {execution_times['Optimized Entropy-Driven Sampling']:.2f} seconds.")
    print(f"Core set size: {X_core.shape[0]} samples ({core_set_size:.2f} KB), Reduction factor: {reduction_factor:.2f}")
    print(f"Full set size: {X_train_full_var.nbytes / 1024:.2f} KB")

    # Check class distribution in the core set
    unique, counts = np.unique(y_core, return_counts=True)
    class_counts = dict(zip(unique, counts))
    print("Class distribution in core set:", class_counts)

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
    # Step 6.5: Further Compression Using K-Means Sampling
    # ============================

    print("\n=== Step 6.5: Further Compression Using K-Means Sampling ===")
    start_time = time.time()

    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin_min

    # Decide on the number of clusters to achieve the desired compression ratio
    desired_compression_ratio = 50  # Adjust as needed
    n_clusters = max(1, len(X_core) // desired_compression_ratio)  # Ensure at least one cluster

    print(f"Number of clusters for desired compression ratio: {n_clusters}")

    # Perform K-Means clustering on the core set
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024, random_state=42)
    kmeans.fit(X_core)

    # Get cluster centers
    cluster_centers = kmeans.cluster_centers_

    # For each cluster, find the closest data point to the cluster center
    closest_data_points_indices, _ = pairwise_distances_argmin_min(cluster_centers, X_core)
    X_core_compressed = X_core[closest_data_points_indices]
    y_core_compressed = y_core.iloc[closest_data_points_indices].reset_index(drop=True)

    # Verify the size of the compressed core set
    print(f"Compressed core set size: {X_core_compressed.shape[0]} samples")

    # Check class distribution in compressed core set
    print("Class counts in compressed core set:")
    print(y_core_compressed.value_counts())

    # Update compression ratio
    reduction_factor_compressed = len(X_train_full_var) / X_core_compressed.shape[0]
    print(f"Compression Ratio after K-Means Sampling: {reduction_factor_compressed:.2f}")

    # Record total execution time for this step
    end_time = time.time()
    execution_times['K-Means Sampling Compression'] = end_time - start_time
    print(f"K-Means sampling compression completed in {execution_times['K-Means Sampling Compression']:.2f} seconds.")

    # ============================
    # Step 6.6: Training Random Forest on K-Means Compressed Core Set
    # ============================

    print("\n=== Step 6.6: Training Random Forest on K-Means Compressed Core Set ===")
    start_time_rf = time.time()

    rf_core_compressed = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        class_weight='balanced_subsample',
        bootstrap=True,
        random_state=42
    )
    print("Training Random Forest on K-Means compressed core set...")
    rf_core_compressed.fit(X_core_compressed, y_core_compressed)

    # Evaluate the model on the test data
    y_pred_core_compressed = rf_core_compressed.predict(X_test_full_var)

    # Adjust target_names based on actual labels present in y_test_full
    labels = np.unique(y_test_full)
    target_names_adjusted = [class_names[int(i)] for i in labels]

    print("\nClassification Report (K-Means Compressed Core Set Model on Test Data):")
    print(classification_report(y_test_full, y_pred_core_compressed, labels=labels, target_names=target_names_adjusted, zero_division=0))

    # Record execution time
    end_time_rf = time.time()
    execution_times['Training on K-Means Compressed Core Set'] = end_time_rf - start_time_rf
    print(f"Random Forest trained and evaluated on K-Means compressed core set in {execution_times['Training on K-Means Compressed Core Set']:.2f} seconds.")

    # =========================
    # Step 7: Comparison of Models
    # =========================
    print("\n=== Step 7: Comparison of Models ===")

    # Evaluate the original Random Forest model on the test data
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
    print(f"K-Means Compressed Core Set Model Accuracy: {rf_core_compressed.score(X_test_full_var, y_test_full):.4f}")

    # =========================
    # Step 8: Statistical Comparison and Summary
    # =========================
    print("\n=== Step 8: Statistical Comparison and Summary ===")

    # Feature count and data size
    full_data_feature_count = X_train_full_var.shape[1]
    core_data_feature_count = X_core.shape[1]
    compressed_core_data_feature_count = X_core_compressed.shape[1]

    summary_df = pd.DataFrame({
        'Dataset': ['Full Data', 'Core Set', 'K-Means Compressed Core'],
        'Samples': [X_train_full_var.shape[0], X_core.shape[0], X_core_compressed.shape[0]],
        'Accuracy': [
            rf_full.score(X_test_full_var, y_test_full),
            rf_core.score(X_test_full_var, y_test_full),
            rf_core_compressed.score(X_test_full_var, y_test_full)
        ],
        'Compression Ratio': [
            1,
            round(reduction_factor, 2),
            round(reduction_factor_compressed, 2)
        ],
        'Data Size (KB)': [
            X_train_full_var.nbytes / 1024,
            X_core.nbytes / 1024,
            X_core_compressed.nbytes / 1024
        ],
        'Number of Features': [
            full_data_feature_count,
            core_data_feature_count,
            compressed_core_data_feature_count
        ]
    })

    print("\n=== Summary Table ===")
    print(summary_df)

    benchmark = Benchmarking(
        X_full=X_train_full_var,
        y_full=y_train_full,
        X_core=X_core,
        y_core=y_core,
        X_compressed=X_core_compressed,
        y_compressed=y_core_compressed,
        X_test=X_test_full_var,
        y_test=y_test_full,
        feature_names=selected_variance_feature_names,
        class_names=class_names
    )

    # =========================
    # Step 9: Statistical Validation of Compression
    # =========================
    print("\n=== Step 9: Statistical Validation of Compression ===")
    start_time = time.time()

    # 1. Compare feature distributions using Kolmogorov-Smirnov test
    print("\nPerforming Kolmogorov-Smirnov tests on feature distributions...")
    ks_results = []
    for i in range(X_train_full_var.shape[1]):
        stat, p_value = ks_2samp(X_train_full_var[:, i], X_core_compressed[:, i])
        ks_results.append(p_value)
    ks_pvalues = np.array(ks_results)

    # Calculate the percentage of features that have similar distributions
    alpha = 0.05  # Significance level
    num_features = X_train_full_var.shape[1]
    num_similar = np.sum(ks_pvalues > alpha)
    print(f"Number of features with similar distributions: {num_similar}/{num_features}")
    print(f"Percentage: {num_similar / num_features * 100:.2f}%")

    # 2. Visualize feature distributions for selected features
    print("\nVisualizing feature distributions for selected features...")
    feature_indices = np.random.choice(range(num_features), size=5, replace=False)
    for idx in feature_indices:
        plt.figure(figsize=(8, 4))
        sns.kdeplot(X_train_full_var[:, idx], label='Full Data', shade=True)
        sns.kdeplot(X_core_compressed[:, idx], label='Compressed Data', shade=True)
        plt.title(f'Distribution Comparison for Feature {selected_variance_feature_names[idx]}')
        plt.legend()
        plt.show()

    # 3. Compare class distributions using Chi-Square test
    print("\nComparing class distributions using Chi-Square test...")
    full_class_counts = y_train_full.value_counts().sort_index()
    compressed_class_counts = y_core_compressed.value_counts().sort_index()
    contingency_table = pd.DataFrame({
        'Full Data': full_class_counts,
        'Compressed Data': compressed_class_counts
    })
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    print(f"Chi-Square Statistic: {chi2:.2f}, p-value: {p_value:.4f}")
    if p_value > alpha:
        print("Class distributions are similar.")
    else:
        print("Class distributions are significantly different.")

    # 4. Visualize data using PCA
    print("\nVisualizing data using PCA...")
    pca = PCA(n_components=2, random_state=42)
    X_full_pca = pca.fit_transform(X_train_full_var)
    X_compressed_pca = pca.transform(X_core_compressed)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_full_pca[:, 0], X_full_pca[:, 1], c=y_train_full, cmap='viridis', alpha=0.1, label='Full Data')
    plt.scatter(X_compressed_pca[:, 0], X_compressed_pca[:, 1], c=y_core_compressed, cmap='viridis', edgecolor='k', label='Compressed Data')
    plt.title('PCA Visualization of Full Data vs. Compressed Data')
    plt.legend()
    plt.show()

    # Record execution time
    end_time = time.time()
    execution_times['Statistical Validation'] = end_time - start_time
    print(f"Statistical validation completed in {execution_times['Statistical Validation']:.2f} seconds.")
    
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
        stat_ks, p_value_ks = ks_2samp(X_train_full_var[:, idx], X_core_compressed[:, idx])
        # Jensen-Shannon Divergence
        hist_full, bin_edges = np.histogram(X_train_full_var[:, idx], bins=50, density=True)
        hist_compressed, _ = np.histogram(X_core_compressed[:, idx], bins=bin_edges, density=True)
        # Add a small value to avoid zeros
        hist_full += 1e-8
        hist_compressed += 1e-8
        js_distance = jensenshannon(hist_full, hist_compressed)
        # Wasserstein Distance
        wasserstein_dist = wasserstein_distance(X_train_full_var[:, idx], X_core_compressed[:, idx])
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
    similarity_logs_df.to_csv(f'feature_similarity_logs_{len(nodes)}.csv', index=False)
    print(f"\nFeature similarity logs saved to 'feature_similarity_logs_{len(nodes)}.csv'.")

    # 3. Display the logs
    print("\nTop Features Similarity Measures:")
    print(similarity_logs_df.head(10))  # Display top 10 for brevity

    # 4. Visualize Distributions of Top Features
    print("\nVisualizing distributions of top important features...")
    for idx in top_feature_indices[:5]:  # Visualize top 5 features
        feature_name = selected_variance_feature_names[idx]
        plt.figure(figsize=(8, 4))
        sns.kdeplot(X_train_full_var[:, idx], label='Full Data', shade=True)
        sns.kdeplot(X_core_compressed[:, idx], label='Compressed Data', shade=True)
        plt.title(f'Distribution Comparison for Feature: {feature_name}')
        plt.legend()
        plt.show()

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
    memory_core = getsizeof(X_core) / 1024 ** 2
    memory_compressed_core = getsizeof(X_core_compressed) / 1024 ** 2
    time_full = execution_times['Optimized Entropy-Driven Sampling'] + execution_times.get('Training on Core Set', 0)
    time_core = execution_times.get('Training on Core Set', 0)
    time_compressed_core = execution_times.get('Training on K-Means Compressed Core Set', 0)

    print(f"Memory Usage - Full Dataset: {memory_full:.2f} MB")
    print(f"Memory Usage - Core Dataset: {memory_core:.2f} MB")
    print(f"Memory Usage - Compressed Core Dataset: {memory_compressed_core:.2f} MB")
    print(f"Training Time - Full Model: {time_full:.2f} seconds")
    print(f"Training Time - Core Model: {time_core:.2f} seconds")
    print(f"Training Time - Compressed Core Model: {time_compressed_core:.2f} seconds")

    # =========================
    # Final Notes
    # =========================
    print("\n=== All Steps Completed ===")
    print("The code now includes K-Means sampling instead of stratified sampling for further compression.")
    
    # End of your script; restore stdout
    tee.close()
    sys.stdout = tee.terminal 


# Loop through the node counts and call the main function for each
if __name__ == '__main__':
    main()
