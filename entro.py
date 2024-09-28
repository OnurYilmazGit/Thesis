import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.special import entr
from collections import defaultdict
from sklearn.metrics import classification_report
from scipy.special import entr
from sklearn.cluster import MiniBatchKMeans

# Import custom classes
from data_loader import DataLoader
from dataPreprocessor import DataPreprocessor
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    # Initialize variables to store execution times
    execution_times = {}

    # Define class names
    class_names = ['Kripke', 'AMG', 'PENNANT', 'linpack', 'LAMMPS', 'Quicksilver']

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
    responses = data_loader.load_responses()
    sensor_data = data_loader.load_sensors()

    # Merge data
    data = pd.merge(sensor_data, responses, on=['Time', 'Node'])

    # Verify loaded nodes
    unique_nodes = data['Node'].unique()
    print("Unique Nodes after Merging:", unique_nodes)
    print("Records per Node:", data['Node'].value_counts())

    # Remove duplicates
    data = data.drop_duplicates()
    print(f"Total records after dropping duplicates: {data.shape[0]}")

    # Map application labels to numerical values
    label_mapping = {
        'Kripke': 0,
        'AMG': 1,
        'PENNANT': 2,
        'linpack': 3,
        'LAMMPS': 4,
        'Quicksilver': 5,
    }

    # Map labels
    data['Value'] = data['Value'].astype(str)  # Ensure labels are strings
    data['Value'] = data['Value'].map(label_mapping)

    # Handle NaN values
    if data['Value'].isnull().any():
        print("Dropping rows with unmapped labels.")
        data.dropna(subset=['Value'], inplace=True)
        data.reset_index(drop=True, inplace=True)

    y = data['Value']
    print("Unique labels after mapping:", y.unique())

    end_time = time.time()
    execution_times['Data Loading and Merging'] = end_time - start_time
    print(f"Data loaded and merged successfully in {execution_times['Data Loading and Merging']:.2f} seconds.")
    print(f"Total records: {data.shape[0]}")

    # Verify no NaN values in y
    if y.isnull().any():
        print("NaN values found in 'y' after handling.")
        return
    else:
        print("No NaN values in 'y' after handling.")

    # ================================
    # Step 2: Time-Based Splitting
    # ================================
    print("\n=== Step 2: Time-Based Splitting ===")
  
    # Sort data by time
    data = data.sort_values('Time').reset_index(drop=True)

    # Determine split index
    split_index = int(len(data) * 0.8)

    # Split data
    data_train = data.iloc[:split_index].reset_index(drop=True)
    data_test = data.iloc[split_index:].reset_index(drop=True)
    print("Training Data Shape:", data_train.shape)
    print("Test Data Shape:", data_test.shape)

    # ================================
    # Step 3: Feature Engineering and Normalization
    # ================================
    print("\n=== Step 3: Feature Engineering and Normalization ===")

    # Apply feature engineering separately
    preprocessor_train = DataPreprocessor(data_train, window_size=20)
    data_train_fe = preprocessor_train.feature_engineering()

    preprocessor_test = DataPreprocessor(data_test, window_size=20)
    data_test_fe = preprocessor_test.feature_engineering()

    # Exclude 'Time' and 'Node' from features
    features_to_exclude = ['Time', 'Node', 'Value']
    # Add any derived features that are based on 'Time' or 'Node' to this list

    # Ensure that the feature columns are the same
    feature_columns_fe = [col for col in data_train_fe.columns if col not in features_to_exclude]
    data_train_fe = data_train_fe[feature_columns_fe + ['Value']]
    data_test_fe = data_test_fe[feature_columns_fe + ['Value']]

    # Update y values after feature engineering and dropping NaNs
    y_train_full = data_train_fe['Value'].reset_index(drop=True)
    y_test_full = data_test_fe['Value'].reset_index(drop=True)

    # Normalize data
    X_train_full = data_train_fe[feature_columns_fe]
    X_test_full = data_test_fe[feature_columns_fe]

    # Handle any remaining NaN values
    X_train_full.fillna(0, inplace=True)
    X_test_full.fillna(0, inplace=True)

    # Fit scaler on training data
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)

    # Transform test data using the same scaler
    X_test_full_scaled = scaler.transform(X_test_full)

    # Record execution time
    end_time = time.time()
    execution_times['Feature Engineering and Normalization'] = end_time - start_time
    print(f"Feature engineering and normalization completed in {execution_times['Feature Engineering and Normalization']:.2f} seconds.")

    # ================================
    # Step 4: Variance Threshold Feature Selection
    # ================================
    print("\n=== Step 4: Variance Threshold Feature Selection ===")
    start_time = time.time()

    # Initialize the variance threshold selector
    variance_threshold = 0.10  # Adjust as needed
    selector = VarianceThreshold(threshold=variance_threshold)

    # Fit and transform the training data
    X_train_full_var = selector.fit_transform(X_train_full_scaled)
    selected_variance_features = selector.get_support(indices=True)
    selected_variance_feature_names = [feature_columns_fe[i] for i in selected_variance_features]

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

    # Step 1: Train Random Forest on the full training data with variance thresholded features 
    rf_full = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42, n_jobs=-1, class_weight='balanced_subsample', min_samples_leaf=4, criterion='entropy', bootstrap=True)
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
    high_uncertainty_threshold = np.percentile(predictive_entropies, 96)
    low_uncertainty_threshold = np.percentile(predictive_entropies, 3)

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

    # Ensure that the core set maintains the same class proportions
    min_samples_per_class = 200  # Minimum samples for each class
    balanced_indices = []
    class_sample_counts = defaultdict(int)

    # Calculate how many samples to include from each class
    total_core_samples = len(selected_indices)
    class_sample_limits = (class_distribution_full * total_core_samples).astype(int)
    class_sample_limits[class_sample_limits < min_samples_per_class] = min_samples_per_class

    print("\nClass sample limits for core set based on full dataset proportions:")
    print(class_sample_limits)

    # Step 5: Create the Core Dataset
    X_core = X_train_full_var[selected_indices]
    y_core = y_train_full.iloc[selected_indices].reset_index(drop=True)

    # Step 6: Print the Class Distribution in the Core Set
    class_distribution_core = y_core.value_counts(normalize=True)
    print("\nClass distribution in core set (proportion):")
    print(class_distribution_core)

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
    rf_core = RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight='balanced_subsample', max_depth=20, bootstrap=True)
    print("Training Random Forest on core set...")
    rf_core.fit(X_core, y_core)

    # Evaluate the core model on the test data with variance-thresholded features
    y_pred_core = rf_core.predict(X_test_full_var)
    print("\nClassification Report (Core Set Model on Test Data):")
    print(classification_report(y_test_full, y_pred_core, target_names=class_names, zero_division=0))

    # Record execution time
    end_time = time.time()
    execution_times['Training on Core Set'] = end_time - start_time
    print(f"Random Forest trained and evaluated on core set in {execution_times['Training on Core Set']:.2f} seconds.")

        
    # ============================
    # Step 6.5: Further Compression Using K-Means Clustering
    # ============================

    print("\n=== Step 6.5: Further Compression Using K-Means Clustering ===")
    start_time = time.time()

    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin_min

    # Decide on the number of clusters to achieve the desired compression ratio
    desired_compression_ratio = 50  # Adjust as needed
    number_of_samples_in_core = X_core.shape[0]
    number_of_clusters = max(1, int(number_of_samples_in_core / desired_compression_ratio))
    print(f"Number of clusters for desired compression ratio: {number_of_clusters}")

    # Apply K-Means clustering to X_core
    print("Applying K-Means clustering to the core set...")
    kmeans = MiniBatchKMeans(n_clusters=number_of_clusters, random_state=42, batch_size=500)
    kmeans.fit(X_core)
    # Find the sample closest to each cluster center
    print("Selecting representative samples from each cluster...")
    closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X_core)
    X_core_kmeans = X_core[closest_indices]
    y_core_kmeans = y_core.iloc[closest_indices].reset_index(drop=True)

    # Verify the size of the compressed core set
    print(f"Compressed core set size: {X_core_kmeans.shape[0]} samples")

    # ============================
    # Step 6.6: Training Random Forest on K-Means Compressed Core Set
    # ============================

    print("\n=== Step 6.6: Training Random Forest on K-Means Compressed Core Set ===")
    start_time_rf = time.time()

    rf_core_kmeans = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        class_weight='balanced_subsample',
        max_depth=20,
        bootstrap=True,
        random_state=42
    )
    print("Training Random Forest on K-Means compressed core set...")
    rf_core_kmeans.fit(X_core_kmeans, y_core_kmeans)
 
    # Evaluate the model on the test data
    y_pred_core_kmeans = rf_core_kmeans.predict(X_test_full_var)
    print("\nClassification Report (K-Means Compressed Core Set Model on Test Data):")
    print(classification_report(y_test_full, y_pred_core_kmeans, target_names=class_names, zero_division=0))

    # Record execution time
    end_time_rf = time.time()
    execution_times['Training on K-Means Compressed Core Set'] = end_time_rf - start_time_rf
    print(f"Random Forest trained and evaluated on K-Means compressed core set in {execution_times['Training on K-Means Compressed Core Set']:.2f} seconds.")

    # Update compression ratio
    reduction_factor_kmeans = len(X_train_full_var) / X_core_kmeans.shape[0]
    print(f"Compression Ratio after K-Means: {reduction_factor_kmeans:.2f}")

    # Record total execution time for this step
    end_time = time.time()
    execution_times['K-Means Compression'] = end_time - start_time
    print(f"K-Means compression and model training completed in {execution_times['K-Means Compression']:.2f} seconds.")


# ---- Surayi bir uopdate et ama obncesinde percentage lardan yuru, 
    # =========================
    # Step 7: Comparison of Models
    # =========================
    print("\n=== Step 7: Comparison of Models ===")

    # Evaluate the original model on the test data with variance-thresholded features
    y_pred_full = rf_full.predict(X_test_full_var)
    print("\nClassification Report (Full Model on Test Data):")
    print(classification_report(y_test_full, y_pred_full, target_names=class_names, zero_division=0))

    # Summarize results
    print("\nSummary of Results:")
    print(f"Full Model Test Accuracy: {rf_full.score(X_test_full_var, y_test_full):.4f}")
    print(f"Core Model Test Accuracy: {rf_core.score(X_test_full_var, y_test_full):.4f}")
    print(f"Compression Ratio: {reduction_factor:.2f}")

    # =========================
    # Step 8: Statistical Comparison and Summary
    # =========================
    print("\n=== Step 8: Statistical Comparison and Summary ===")

    # Feature count and data size
    full_data_feature_count = X_train_full_var.shape[1]
    core_data_feature_count = X_core.shape[1]
    kmeans_core_data_feature_count = X_core_kmeans.shape[1]

    summary_df = pd.DataFrame({
        'Dataset': ['Full Data', 'Core Set', 'KMean Core '],
        'Samples': [X_train_full_var.shape[0], X_core.shape[0], X_core_kmeans.shape[0]],
        'Accuracy': [
            rf_full.score(X_test_full_var, y_test_full),
            rf_core.score(X_test_full_var, y_test_full),
            rf_core_kmeans.score(X_test_full_var, y_test_full)
        ],
        'Compression Ratio': [
            1,
            round(reduction_factor, 2),
            round(reduction_factor_kmeans, 2)
        ],
        'Data Size (KB)': [
            X_train_full_var.nbytes / 1024,
            X_core.nbytes / 1024,
            X_core_kmeans.nbytes / 1024
        ],
        'Number of Features': [
            full_data_feature_count,
            core_data_feature_count,
            kmeans_core_data_feature_count
        ]
    })

    print("\n=== Summary Table ===")
    print(summary_df)



if __name__ == '__main__':
    main()
