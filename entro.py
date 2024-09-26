import pandas as pd
import numpy as np
import os
import warnings
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import _tree
from benchmarking import evaluate_model


# Import custom classes
from data_loader import DataLoader
from dataPreprocessor import DataPreprocessor

# Suppress warnings
warnings.filterwarnings("ignore")

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
    nodes = [f'node{i}' for i in range(4)]
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

    # ============================================
    # Step 3: Feature Engineering and Normalization
    # ============================================
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

    # ==========================================
    # Step 4: Entropy-Driven Sampling Strategy
    # ==========================================
    print("\n=== Step 4: Entropy-Driven Sampling Strategy ===")
    start_time = time.time()

    # Train a Random Forest on the full training data
    rf_full = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight='balanced_subsample', min_samples_leaf=2)
    print("Max depth of Random Forest:", rf_full.max_depth, "Number of estimators:", rf_full.n_estimators)
    rf_full.fit(X_train_full_scaled, y_train_full)

    # Extract high-entropy samples using a percentile threshold
    preprocessor = DataPreprocessor(None)
    percentile = 99  # Adjust this value to increase compression ratio
    high_entropy_indices = preprocessor.get_high_entropy_samples(
        rf_full, X_train_full_scaled, y_train_full, percentile=percentile)

    # Create core set from high-entropy samples
    X_core = X_train_full_scaled[high_entropy_indices]
    y_core = y_train_full.iloc[high_entropy_indices].reset_index(drop=True)

    core_set_size = X_core.nbytes / 1024  # in KB
    reduction_factor = len(X_train_full_scaled) / len(X_core)

    end_time = time.time()
    execution_times['Entropy-Driven Sampling'] = end_time - start_time
    print(f"Entropy-driven sampling completed in {execution_times['Entropy-Driven Sampling']:.2f} seconds.")
    print(f"Core set size: {X_core.shape[0]} samples ({core_set_size:.2f} KB), Reduction factor: {reduction_factor:.2f}")
    print(f"Full set size: {X_train_full_scaled.nbytes / 1024:.2f} KB")

    # Check class distribution in the core set
    unique, counts = np.unique(y_core, return_counts=True)
    class_counts = dict(zip(unique, counts))
    print("Class distribution in core set:", class_counts)

    # =============================================
    # Step 5: Training Random Forest on Core Set Data
    # =============================================
    start_time = time.time()
    print("\n=== Step 5: Training Random Forest on Core Set Data ===")

    # Train Random Forest on the core set
    rf_core = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight='balanced_subsample')
    rf_core.fit(X_core, y_core)

    # Evaluate the core model on the test data
    y_pred_core = rf_core.predict(X_test_full_scaled)
    print("\nClassification Report (Core Set Model on Test Data):")
    print(classification_report(y_test_full, y_pred_core, target_names=class_names))

    # Record execution time
    end_time = time.time()
    execution_times['Training on Core Set'] = end_time - start_time
    print(f"Random Forest trained and evaluated on core set in {execution_times['Training on Core Set']:.2f} seconds.")

    # =========================
    # Step 6: Comparison of Models
    # =========================
    print("\n=== Step 6: Comparison of Models ===")

    # Evaluate the original model on the test data
    y_pred_full = rf_full.predict(X_test_full_scaled)
    print("\nClassification Report (Full Model on Test Data):")
    print(classification_report(y_test_full, y_pred_full, target_names=class_names))

    # Summarize results
    print("\nSummary of Results:")
    print(f"Full Model Test Accuracy: {rf_full.score(X_test_full_scaled, y_test_full):.4f}")
    print(f"Core Model Test Accuracy: {rf_core.score(X_test_full_scaled, y_test_full):.4f}")
    print(f"Compression Ratio: {reduction_factor:.2f}")

    # =========================
    # Step 7: Statistical Comparison
    # =========================
    print("\n=== Step 7: Statistical Comparison and Summary ===")

    # Feature count and data size
    full_data_feature_count = X_train_full_scaled.shape[1]
    core_data_feature_count = X_core.shape[1]

    summary_df = pd.DataFrame({
        'Dataset': ['Full Data', 'Core Set'],
        'Samples': [X_train_full_scaled.shape[0], X_core.shape[0]],
        'Accuracy': [rf_full.score(X_test_full_scaled, y_test_full), rf_core.score(X_test_full_scaled, y_test_full)],
        'Compression Ratio': [1, round(reduction_factor, 2)],
        'Data Size (KB)': [X_train_full_scaled.nbytes / 1024, X_core.nbytes / 1024],
        'Number of Features': [full_data_feature_count, core_data_feature_count]
    })

    print("\n=== Summary Table ===")
    print(summary_df)

if __name__ == '__main__':
    main()
