import pandas as pd
import numpy as np
import time
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.special import entr
from collections import defaultdict
from sklearn.metrics import classification_report
from benchmarking2 import Benchmarking
from data_loader import DataLoader

# Suppress warnings
warnings.filterwarnings("ignore")

def main():
    # Initialize variables to store execution times
    execution_times = {}

    # Define class names, including 'None' as the 7th label
    class_names = ['Kripke', 'AMG', 'PENNANT', 'linpack', 'LAMMPS', 'Quicksilver', 'None']

    # ================================
    # Step 1: Data Loading and Merging
    # ================================
    start_time = time.time()
    print("\n=== Step 1: Data Loading and Merging ===")

    responses_path = '../responses'
    sensors_path = '../sensors'
    nodes = [f'node{i}' for i in range(16)]
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

    # ================================
    # Step 2: Time-Based Splitting
    # ================================
    print("\n=== Step 2: Time-Based Splitting ===")

    # Sort data by time
    data_final = data_final.sort_values('Time').reset_index(drop=True)

    # Determine split index
    split_index = int(len(data_final) * 0.8)

    # Split data
    data_train = data_final.iloc[:split_index].reset_index(drop=True)
    data_test = data_final.iloc[split_index:].reset_index(drop=True)
    print("Training Data Shape:", data_train.shape)
    print("Test Data Shape:", data_test.shape)

    # ================================
    # Step 3: Data Preparation and Normalization
    # ================================
    print("\n=== Step 3: Data Preparation and Normalization ===")

    # Exclude 'Time' from features
    features_to_exclude = ['Time', 'Value']
    # Add any derived features that are based on 'Time' to this list

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

    # Record execution time
    end_time = time.time()
    execution_times['Data Preparation and Normalization'] = end_time - start_time
    print(f"Data preparation and normalization completed in {execution_times['Data Preparation and Normalization']:.2f} seconds.")

    # ================================
    # Step 4: Variance Threshold Feature Selection
    # ================================
    print("\n=== Step 4: Variance Threshold Feature Selection ===")
    start_time = time.time()

    # Initialize the variance threshold selector
    variance_threshold = 0.1  # Adjust as needed
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

    # Record execution time
    end_time = time.time()
    execution_times['Variance Threshold Feature Selection'] = end_time - start_time
    print(f"Variance Threshold feature selection completed in {execution_times['Variance Threshold Feature Selection']:.2f} seconds.")

    # ================================
    # Optimized Entropy-Driven Sampling Strategy with Predictive Uncertainties
    # ================================
    print("\n=== Optimized Entropy-Driven Sampling Strategy ===")
    start_time = time.time()

    # Step 1: Train Random Forest on the full training data with variance-thresholded features
    rf_full = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced_subsample',
        min_samples_leaf=5,
        criterion='entropy',
        bootstrap=True
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
    # Step 6: Training Random Forest on Core Set Data
    # ============================
    start_time = time.time()
    print("\n=== Step 6: Training Random Forest on Core Set Data ===")

    # Train Random Forest on the core set
    rf_core = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        class_weight='balanced_subsample',
        max_depth=20,
        bootstrap=True
    )
    print("Training Random Forest on core set...")
    rf_core.fit(X_core, y_core)

    # Evaluate the core model on the test data with variance-thresholded features
    y_pred_core = rf_core.predict(X_test_full_var)

    # Adjust target_names based on actual labels present in y_test_full
    labels = np.unique(y_test_full)
    target_names_adjusted = [class_names[int(i)] for i in labels]

    print("\nClassification Report (Core Set Model on Test Data):")
    print(classification_report(y_test_full, y_pred_core, labels=labels, target_names=target_names_adjusted, zero_division=0))

    # Record execution time
    end_time = time.time()
    execution_times['Training on Core Set'] = end_time - start_time
    print(f"Random Forest trained and evaluated on core set in {execution_times['Training on Core Set']:.2f} seconds.")

    # ============================
    # Step 6.5: Further Compression Using Stratified Sampling
    # ============================

    print("\n=== Step 6.5: Further Compression Using Stratified Sampling ===")
    start_time = time.time()

    from sklearn.model_selection import train_test_split

    # Decide on the sampling fraction to achieve the desired compression ratio
    desired_compression_ratio = 50  # Adjust as needed
    sampling_fraction = 1 / desired_compression_ratio  # e.g., 1/50 = 0.02

    print(f"Sampling fraction for desired compression ratio: {sampling_fraction:.4f}")

    # Perform stratified sampling on the core set
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

    # Record total execution time for this step
    end_time = time.time()
    execution_times['Stratified Sampling Compression'] = end_time - start_time
    print(f"Stratified sampling compression completed in {execution_times['Stratified Sampling Compression']:.2f} seconds.")

    # ============================
    # Step 6.6: Training Random Forest on Stratified Compressed Core Set
    # ============================

    print("\n=== Step 6.6: Training Random Forest on Stratified Compressed Core Set ===")
    start_time_rf = time.time()

    rf_core_sampled = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        class_weight='balanced_subsample',
        max_depth=20,
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

    # =========================
    # Step 7: Comparison of Models
    # =========================
    print("\n=== Step 7: Comparison of Models ===")

    # Evaluate the original model on the test data with variance-thresholded features
    y_pred_full = rf_full.predict(X_test_full_var)

    # Adjust target_names based on actual labels present in y_test_full
    labels = np.unique(y_test_full)
    target_names_adjusted = [class_names[int(i)] for i in labels]

    print("\nClassification Report (Full Model on Test Data):")
    print(classification_report(y_test_full, y_pred_full, labels=labels, target_names=target_names_adjusted, zero_division=0))

    # Summarize results
    print("\nSummary of Results:")
    print(f"Full Model Test Accuracy: {rf_full.score(X_test_full_var, y_test_full):.4f}")
    print(f"Core Model Test Accuracy: {rf_core.score(X_test_full_var, y_test_full):.4f}")
    print(f"Compressed Core Model Test Accuracy: {rf_core_sampled.score(X_test_full_var, y_test_full):.4f}")
    print(f"Compression Ratio: {reduction_factor:.2f}")
    print(f"Compression Ratio after Stratified Sampling: {reduction_factor_sampled:.2f}")

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

    benchmark = Benchmarking(
        X_full=X_train_full_var,
        y_full=y_train_full,
        X_core=X_core,
        y_core=y_core,
        X_compressed=X_core_sampled,
        y_compressed=y_core_sampled,
        X_test=X_test_full_var,
        y_test=y_test_full,
        feature_names=selected_variance_feature_names,
        class_names=class_names
    )

    # Perform benchmarking
    #benchmark.check_data_leakage()
    #benchmark.clustering_evaluation()
    #benchmark.compare_model_performance(rf_full, rf_core, rf_core_sampled)
    #benchmark.statistical_similarity_tests()
    #benchmark.feature_importance_comparison(rf_full, rf_core)
    #benchmark.visualize_feature_distributions()
    #benchmark.cross_validation_checks(rf_core)

if __name__ == '__main__':
    main()
