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
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import warnings

# Import custom classes
from data_loader import DataLoader
from dataPreprocessor import DataPreprocessor

# Suppress warnings
warnings.filterwarnings("ignore")


def apply_kmeans_per_class(X, y, samples_per_class):
    X_core = []
    y_core = []
    unique_classes = np.unique(y)
    
    for cls in unique_classes:
        # Get all samples of the current class
        X_cls = X[y == cls]
        y_cls = y[y == cls]
        
        # If the number of samples is less than or equal to desired samples, take all
        if len(X_cls) <= samples_per_class:
            X_core.append(X_cls)
            y_core.append(y_cls)
            continue
        
        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=samples_per_class, random_state=42)
        kmeans.fit(X_cls)
        cluster_centers = kmeans.cluster_centers_
        
        # Find the nearest sample to each cluster center
        distances = cdist(cluster_centers, X_cls)
        nearest_indices = np.argmin(distances, axis=1)
        X_selected = X_cls[nearest_indices]
        y_selected = y_cls.iloc[nearest_indices]
        
        # Append to core set
        X_core.append(X_selected)
        y_core.append(y_selected)
    
    # Concatenate all classes
    X_core = np.vstack(X_core)
    y_core = np.concatenate(y_core)
    
    return X_core, y_core


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
    variance_threshold = 0.25  # Adjust as needed
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
    # Apply K-Means Clustering per Class
    # ================================
    print("\n=== Applying K-Means Clustering per Class ===")
    start_time = time.time()

    samples_per_class = 2000  # Adjust to achieve desired compression
    X_core_kmeans, y_core_kmeans = apply_kmeans_per_class(X_train_full_var, y_train_full, samples_per_class)

    # Check compression ratio
    reduction_factor_kmeans = len(X_train_full_var) / len(X_core_kmeans)
    print(f"Compression Ratio with K-Means: {reduction_factor_kmeans:.2f}")

    # Record execution time
    end_time = time.time()
    execution_times['K-Means Clustering'] = end_time - start_time
    print(f"K-Means clustering completed in {execution_times['K-Means Clustering']:.2f} seconds.")

    # ============================
    # Step 6: Training Random Forest on K-Means Core Set Data
    # ============================
    start_time = time.time()
    print("\n=== Step 6: Training Random Forest on K-Means Core Set Data ===")

    # Train Random Forest on the K-Means core set
    rf_core_kmeans = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        class_weight='balanced_subsample',
        max_depth=20,
        bootstrap=True,
        random_state=42
    )
    print("Training Random Forest on K-Means core set...")
    rf_core_kmeans.fit(X_core_kmeans, y_core_kmeans)

    # Evaluate the model
    y_pred_core_kmeans = rf_core_kmeans.predict(X_test_full_var)
    print("\nClassification Report (K-Means Core Set Model on Test Data):")
    print(classification_report(y_test_full, y_pred_core_kmeans, target_names=class_names, zero_division=0))

    # Record execution time
    end_time = time.time()
    execution_times['Training on K-Means Core Set'] = end_time - start_time
    print(f"Random Forest trained and evaluated on K-Means core set in {execution_times['Training on K-Means Core Set']:.2f} seconds.")

    # Summarize results
    core_kmeans_accuracy = rf_core_kmeans.score(X_test_full_var, y_test_full)
    print(f"K-Means Core Model Test Accuracy: {core_kmeans_accuracy:.4f}")

    # =========================
    # Step 7: Comparison of Models
    # =========================
    print("\n=== Step 7: Comparison of Models ===")

    # Evaluate the original model on the test data with variance-thresholded features
    y_pred_full = rf_full.predict(X_test_full_var)
    print("\nClassification Report (Full Model on Test Data):")
    print(classification_report(y_test_full, y_pred_full, target_names=class_names, zero_division=0))

    # Summarize results
    full_model_accuracy = rf_full.score(X_test_full_var, y_test_full)
    print("\nSummary of Results:")
    print(f"Full Model Test Accuracy: {full_model_accuracy:.4f}")
    print(f"K-Means Core Model Test Accuracy: {core_kmeans_accuracy:.4f}")
    print(f"Compression Ratio with K-Means: {reduction_factor_kmeans:.2f}")

    # =========================
    # Step 8: Statistical Comparison and Summary
    # =========================
    print("\n=== Step 8: Statistical Comparison and Summary ===")

    # Feature count and data size
    full_data_feature_count = X_train_full_var.shape[1]
    core_data_feature_count = X_core_kmeans.shape[1]

    summary_df = pd.DataFrame({
        'Dataset': ['Full Data', 'K-Means Core Set'],
        'Samples': [X_train_full_var.shape[0], X_core_kmeans.shape[0]],
        'Accuracy': [full_model_accuracy, core_kmeans_accuracy],
        'Compression Ratio': [1, round(reduction_factor_kmeans, 2)],
        'Data Size (KB)': [X_train_full_var.nbytes / 1024, X_core_kmeans.nbytes / 1024],
        'Number of Features': [full_data_feature_count, core_data_feature_count]
    })

    print("\n=== Summary Table ===")
    print(summary_df)


if __name__ == '__main__':
    main()
