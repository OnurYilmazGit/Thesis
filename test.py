import os
import warnings
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

# Import custom classes
from data_loader import DataLoader
from dataPreprocessor import DataPreprocessor

# Import benchmarking functions
from benchmarking import (plot_learning_curve, evaluate_model, perform_cross_validation, statistical_comparison)

import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings("ignore")


def visualize_lag_diff_features(data_train_fe, target_column='Value'):
    """ Visualize lag and difference features for a given dataset. """
    time_column = 'Time'  # Assuming 'Time' column exists in the data

    # Create lag and difference features for visualization
    data_train_fe['Lag 1'] = data_train_fe[target_column].shift(1)
    data_train_fe['Diff 1'] = data_train_fe[target_column].diff()

    # Plot the original target variable, lag, and difference features
    plt.figure(figsize=(12, 6))

    # Plot target variable and lag feature
    plt.subplot(2, 1, 1)
    plt.plot(data_train_fe[time_column], data_train_fe[target_column], label=f'{target_column}', marker='o')
    plt.plot(data_train_fe[time_column], data_train_fe['Lag 1'], label=f'Lag 1 ({target_column} at t-1)', linestyle='--', marker='x')
    plt.title(f'{target_column} and Lag 1 Feature')
    plt.xlabel('Time')
    plt.ylabel(target_column)
    plt.legend()

    # Plot difference feature
    plt.subplot(2, 1, 2)
    plt.plot(data_train_fe[time_column], data_train_fe['Diff 1'], label=f'Difference ({target_column} Change)', color='green', marker='o')
    plt.title('Difference Feature (Value Change)')
    plt.xlabel('Time')
    plt.ylabel(f'{target_column} Difference')
    plt.legend()

    plt.tight_layout()
    plt.show()

def visualize_lag_diff_features_multiple(data, target_column='Value', time_column='Time', max_lag=5):
    """ Visualize lag and difference features for multiple lags for a given dataset. """
    
    plt.figure(figsize=(12, 10))

    # Plot target variable and lag features
    for lag in range(1, max_lag + 1):
        data[f'Lag {lag}'] = data[target_column].shift(lag)
        data[f'Diff {lag}'] = data[target_column].diff(lag)

        # Plot original value vs lag
        plt.subplot(2, 1, 1)
        plt.plot(data[time_column], data[target_column], label=f'{target_column}', marker='o')
        plt.plot(data[time_column], data[f'Lag {lag}'], label=f'Lag {lag} (t-{lag})', linestyle='--', marker='x')
    
    plt.title(f'{target_column} and Multiple Lag Features (Up to Lag {max_lag})')
    plt.xlabel('Time')
    plt.ylabel(target_column)
    plt.legend()

    # Plot differences
    plt.subplot(2, 1, 2)
    for lag in range(1, max_lag + 1):
        plt.plot(data[time_column], data[f'Diff {lag}'], label=f'Difference (t-{lag})', linestyle='-', marker='o')
    
    plt.title(f'Difference Features (Up to Lag {max_lag})')
    plt.xlabel('Time')
    plt.ylabel(f'{target_column} Difference')
    plt.legend()

    plt.tight_layout()
    plt.show()

def compare_lag_features(full_data, core_data, target_column='Value', max_lag=5):
    comparisons = {}
    
    for lag in range(1, max_lag + 1):
        full_lag_col = f'Lag {lag}'
        core_lag_col = f'Lag {lag}'

        # Compute Mean Absolute Error (MAE) between lag features in full and core set
        mae = np.mean(np.abs(full_data[full_lag_col] - core_data[core_lag_col]))
        comparisons[f'Lag {lag} MAE'] = mae

        # Compute Correlation between the lag features
        correlation = full_data[full_lag_col].corr(core_data[core_lag_col])
        comparisons[f'Lag {lag} Correlation'] = correlation
    
    comparison_df = pd.DataFrame(list(comparisons.items()), columns=['Metric', 'Value'])
    return comparison_df
def add_lag_diff_features(data, target_column='Value', max_lag=5):
    """ Adds lag and difference features to the dataset for the specified target column. """
    for lag in range(1, max_lag + 1):
        data[f'Lag {lag}'] = data[target_column].shift(lag)
        data[f'Diff {lag}'] = data[target_column].diff(lag)
    return data

def compare_lag_features(full_data, core_data, target_column='Value', max_lag=5):
    comparisons = {}
    
    for lag in range(1, max_lag + 1):
        full_lag_col = f'Lag {lag}'
        core_lag_col = f'Lag {lag}'

        # Compute Mean Absolute Error (MAE) between lag features in full and core set
        mae = np.mean(np.abs(full_data[full_lag_col] - core_data[core_lag_col]))
        comparisons[f'Lag {lag} MAE'] = mae

        # Compute Correlation between the lag features
        correlation = full_data[full_lag_col].corr(core_data[core_lag_col])
        comparisons[f'Lag {lag} Correlation'] = correlation
    
    comparison_df = pd.DataFrame(list(comparisons.items()), columns=['Metric', 'Value'])
    return comparison_df

def stratified_sampling_core_set(X_train_full, y_train_full, compression_ratio=0.005):
    """
    Perform stratified sampling on the training data to create a core set.
    
    Parameters:
    - X_train_full: Full training data
    - y_train_full: Labels of the training data
    - compression_ratio: The ratio by which the dataset is compressed
    
    Returns:
    - X_core: Core set of training data
    - y_core: Core set labels
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=compression_ratio, random_state=42)
    
    for core_index, _ in sss.split(X_train_full, y_train_full):
        X_core = X_train_full[core_index]
        y_core = y_train_full[core_index]
    
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
    nodes = [f'node{i}' for i in range(10)] 
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

    # ============================================
    # Step 2: Splitting Data Before Feature Engineering
    # ============================================
    start_time = time.time()
    print("\n=== Step 2: Splitting Data Before Feature Engineering ===")

    # Extract indices
    indices = np.arange(len(data))

    # Split data indices before any preprocessing
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y)
    
    # Create training and test dataframes
    data_train = data.iloc[train_indices].reset_index(drop=True)
    data_test = data.iloc[test_indices].reset_index(drop=True)

    # ============================================
    # Step 3: Feature Engineering and Normalization
    # ============================================
    print("\n=== Step 3: Feature Engineering and Normalization ===")
    
    # Apply feature engineering separately
    preprocessor_train = DataPreprocessor(data_train, window_size=20)
    data_train_fe = preprocessor_train.feature_engineering()
    
    preprocessor_test = DataPreprocessor(data_test, window_size=20)
    data_test_fe = preprocessor_test.feature_engineering()
    
    # Ensure that the feature columns are the same
    feature_columns_fe = [col for col in data_train_fe.columns if col not in ['Time', 'Node', 'Value']]
    data_test_fe = data_test_fe[feature_columns_fe + ['Value']]
    
    # Update y values after feature engineering and dropping NaNs
    y_train_full = data_train_fe['Value'].reset_index(drop=True)
    y_test_full = data_test_fe['Value'].reset_index(drop=True)
    
    # Normalize data
    X_train_full = data_train_fe[feature_columns_fe]
    X_test_full = data_test_fe[feature_columns_fe]
    
    # Fit scaler on training data
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    
    # Transform test data using the same scaler
    X_test_full_scaled = scaler.transform(X_test_full)


    # ==========================================
    # Step 4: Stratified Sampling for Core Set Selection
    # ==========================================
    print("\n=== Step 4: Selecting Core Set using Stratified Sampling ===")
    
    X_core, y_core = stratified_sampling_core_set(X_train_full_scaled, y_train_full, compression_ratio=0.005)
    
    print(f"Core Set Size: {X_core.shape[0]} samples")

    # Visualize for core set
    core_set_df = pd.DataFrame(X_core, columns=feature_columns_fe[:X_core.shape[1]])
    core_set_df['Value'] = y_core
    core_set_df['Time'] = data_train_fe['Time'].iloc[core_set_df.index]  # Assuming Time is in the original dataset

    # Add lag and difference features to both full and core datasets
    print("Adding lag and difference features...")
    data_train_fe = add_lag_diff_features(data_train_fe, target_column='Value', max_lag=5)
    core_set_df = add_lag_diff_features(core_set_df, target_column='Value', max_lag=5)

    print("Visualizing for Core Set:")
    visualize_lag_diff_features_multiple(core_set_df, target_column='Value', max_lag=5)

    # Visualize for full dataset
    print("Visualizing for Full Dataset:")
    visualize_lag_diff_features_multiple(data_train_fe, target_column='Value', max_lag=5)

    # Perform comparison and print results
    core_comparison_df = compare_lag_features(data_train_fe, core_set_df, target_column='Value', max_lag=5)
    print("\nComparison between Full Dataset and Core Set Lag Features:")
    print(core_comparison_df)

    # =========================
    # Step 10: Additional Evaluations (Optional)
    # =========================
    # For example, feature importance analysis, error analysis, etc.
    # You can add these steps as needed.

if __name__ == '__main__':
    main()
