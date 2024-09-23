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
    print("Training Data Shape:", data_train.head())
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

    # ============================================
    # Step 3.1: Visualize Lag and Difference Features
    # ============================================
    print("\n=== Step 3.1: Visualizing Lag and Difference Features ===")
    visualize_lag_diff_features(data_train_fe, target_column='Value')

    # Verify that all nodes are present after feature engineering
    feature_nodes_train = set([feature.split('.')[0] for feature in feature_columns_fe])
    print("Nodes represented in training features:", feature_nodes_train)
    
    feature_nodes_test = set([feature.split('.')[0] for feature in feature_columns_fe])
    
    # Record execution time
    end_time = time.time()
    execution_times['Feature Engineering and Normalization'] = end_time - start_time
    print(f"Feature engineering and normalization completed in {execution_times['Feature Engineering and Normalization']:.2f} seconds.")
    # ==========================================
    # Step 4: Feature Selection using Variance Threshold
    # ==========================================
    print("\n=== Step 4: Feature Selection using Variance Threshold ===")
    start_time = time.time()

    # Apply Variance Threshold for feature selection
    selector = VarianceThreshold(threshold=0.1)
    X_train_full_scaled_fs = selector.fit_transform(X_train_full_scaled)
    X_test_full_scaled_fs = selector.transform(X_test_full_scaled)

    # Update feature columns based on the selected features
    selected_features = selector.get_support(indices=True)
    feature_columns = [feature_columns_fe[i] for i in selected_features]

    # Print feature count before and after selection
    print(f"Old Feature Count: {X_train_full_scaled.shape[1]}")
    print(f"New Feature Count: {X_train_full_scaled_fs.shape[1]}")

    # Record execution time
    end_time = time.time()
    execution_times['Variance Threshold Feature Selection'] = end_time - start_time
    print(f"Variance Threshold feature selection completed in {execution_times['Variance Threshold Feature Selection']:.2f} seconds.")

    # ==========================================
    # Step 5: Training Random Forest on Selected Data
    # ==========================================
    start_time = time.time()
    print("\n=== Step 5: Training Random Forest on Selected Data ===")

    # Train Random Forest on the selected training data
    rf_selected = RandomForestClassifier(max_depth=200, n_jobs=-1, random_state=42)
    rf_selected.fit(X_train_full_scaled_fs, y_train_full)

    # Evaluate the full model on the selected test data
    metrics_selected = evaluate_model(
        rf_selected, X_test_full_scaled_fs, y_test_full, class_names,
        model_name="Random Forest (Selected Data)",
        filename_prefix="rf_selected_data"
    )
    accuracy_selected = metrics_selected['accuracy']

    print(f"Accuracy of Random Forest after feature selection: {accuracy_selected:.4f}")

    end_time = time.time()
    execution_times['Random Forest (Selected Data)'] = end_time - start_time
    print(f"Random Forest trained and evaluated in {execution_times['Random Forest (Selected Data)']:.2f} seconds.")

    # ==========================================
    # Step 5.1: Feature Importance Analysis
    # ==========================================
    print("\n=== Step 5.1: Feature Importance Analysis ===")

    # Get feature importances from the model
    importances = rf_selected.feature_importances_

    # Create a DataFrame for feature importances
    feature_importances = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importances
    })

    # Display top 20 features
    top_features = feature_importances.sort_values(by='Importance', ascending=False).head(20)
    print("\nTop 20 Feature Importances:")
    print(top_features)

    # Save feature importances to a CSV file for reference
    feature_importances.to_csv('feature_importances.csv', index=False)

    # Function to map feature names to their source CSV files
    def map_feature_to_csv(feature_name):
        # Assuming feature names are in the format 'nodeX.metric' or 'nodeX.metric_lag_Y'
        # Extract the sensor name before the first '_lag_' or '_diff_' or other suffix
        return feature_name.split('_lag_')[0].split('_diff_')[0]

    # Apply the mapping to all features
    feature_importances['Source_CSV'] = feature_importances['Feature'].apply(map_feature_to_csv)

    # Aggregate importances by CSV file
    csv_importances = feature_importances.groupby('Source_CSV')['Importance'].sum().reset_index()

    # Sort CSV files by aggregated importance
    csv_importances_sorted = csv_importances.sort_values(by='Importance', ascending=False)

    # Display the top 10 CSV files contributing to the model
    print("\nTop 10 CSV Files by Aggregated Feature Importance:")
    print(csv_importances_sorted.head(10))

    # Save CSV importances to a CSV file for reference
    csv_importances_sorted.to_csv('csv_importances.csv', index=False)

    # Plot Top 20 Features
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig('top_20_feature_importances.png')
    plt.show()

    # Plot Top 10 CSV Files
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Source_CSV', data=csv_importances_sorted.head(10))
    plt.title('Top 10 CSV Files by Aggregated Feature Importance')
    plt.tight_layout()
    plt.savefig('top_10_csv_importances.png')
    plt.show()

    # ==========================================
    # Step 6: Selecting Core Set based on Stratified Sampling
    # ==========================================
    start_time = time.time()
    print("\n=== Step 6: Selecting Core Set based on Stratified Sampling ===")

    compression_ratio = 0.005  # Adjust as needed

    # Select the core set from the training data only
    preprocessor = DataPreprocessor(data_train_fe, window_size=20)  # Re-instantiate if needed
    core_set, X_core, y_core = preprocessor.select_core_set_by_rf(
        X_train_full_scaled_fs, y_train_full, data_train_fe.iloc[:, :], compression_ratio=compression_ratio)

    core_set_size = X_core.nbytes / 1024  # in KB
    reduction_factor = len(X_train_full_scaled_fs) / len(X_core)

    end_time = time.time()
    execution_times['Core Set Selection'] = end_time - start_time
    print(f"Core set selection completed in {execution_times['Core Set Selection']:.2f} seconds.")
    print(f"Core set size: {X_core.shape[0]} samples ({core_set_size:.2f} KB), Reduction factor: {reduction_factor:.2f}")

    # Check class distribution in the core set
    unique, counts = np.unique(y_core, return_counts=True)
    class_counts = dict(zip(unique, counts))
    print("Class distribution in core set:", class_counts)

    # =============================================
    # Step 7: Training Random Forest on Core Set Data
    # =============================================
    start_time = time.time()
    print("\n=== Step 7: Training Random Forest on Core Set Data ===")

    X_train_core, X_test_core, y_train_core, y_test_core = train_test_split(
        X_core, y_core, test_size=0.2, random_state=42, stratify=y_core)

    rf_core = RandomForestClassifier(max_depth=200, n_jobs=-1, random_state=42)
    rf_core.fit(X_train_core, y_train_core)

    # Evaluate the core model on the core test data
    metrics_core = evaluate_model(
        rf_core, X_test_core, y_test_core, class_names,
        model_name="Random Forest (Core Set)",
        filename_prefix="rf_core_data"
    )
    accuracy_core = metrics_core['accuracy']

    end_time = time.time()
    execution_times['Random Forest (Core Set)'] = end_time - start_time
    print(f"Random Forest trained and evaluated on core set in {execution_times['Random Forest (Core Set)']:.2f} seconds.")

    # ==========================================
    # Step 7.1: Visualizing Lag and Difference Features for Core Set and Full Dataset
    # ==========================================
    print("\n=== Step 7.1: Visualizing Lag and Difference Features for Full Dataset and Core Set ===")

    # Visualize for full dataset
    print("Visualizing for Full Dataset:")
    visualize_lag_diff_features_multiple(data_train_fe, target_column='Value', max_lag=5)

    # Visualize for core set
    core_set_df = core_set.copy()
    core_set_df['Time'] = data_train_fe['Time'].iloc[core_set.index]  # Assuming Time is in the original dataset
    print("Visualizing for Core Set:")
    visualize_lag_diff_features_multiple(core_set_df, target_column='Value', max_lag=5)

    # Perform comparison and print results
    core_comparison_df = compare_lag_features(data_train_fe, core_set_df, target_column='Value', max_lag=5)
    print("\nComparison between Full Dataset and Core Set Lag Features:")
    print(core_comparison_df)


    # =============================================
    # Step 8: Evaluating Core Set Model on Full Test Data
    # =============================================
    print("\n=== Step 8: Evaluating Core Set Model on Full Test Data ===")

    # Evaluate the core model on the same test set as the full model
    metrics_full_core = evaluate_model(
        rf_core, X_test_full_scaled_fs, y_test_full, class_names,
        model_name="Core Set Model on Full Test Data",
        filename_prefix="core_model_on_full_test"
    )
    accuracy_full_core = metrics_full_core['accuracy']
    print(f"Accuracy of Core Model on Full Test Data: {accuracy_full_core:.4f}")

    # =========================
    # Step 9: Statistical Comparison and Summary of Results
    # =========================
    print("\n=== Step 9: Statistical Comparison and Summary of Results ===")

    # Calculate feature counts
    full_data_feature_count = X_train_full_scaled_fs.shape[1]
    core_data_feature_count = X_core.shape[1]

    summary_df = pd.DataFrame({
        'Dataset': ['Full Data', 'Core Set', 'Core Model on Full Test Data'],
        'Samples': [X_train_full_scaled_fs.shape[0], X_core.shape[0], X_test_full_scaled_fs.shape[0]],
        'Accuracy': [metrics_selected['accuracy'], metrics_core['accuracy'], metrics_full_core['accuracy']],
        'Precision': [metrics_selected['precision'], metrics_core['precision'], metrics_full_core['precision']],
        'Recall': [metrics_selected['recall'], metrics_core['recall'], metrics_full_core['recall']],
        'F1 Score': [metrics_selected['f1_score'], metrics_core['f1_score'], metrics_full_core['f1_score']],
        'Data Size (KB)': [X_train_full_scaled_fs.nbytes / 1024, core_set_size, X_test_full_scaled_fs.nbytes / 1024],
        'Number of Features': [full_data_feature_count, core_data_feature_count, full_data_feature_count]
    })

    print(summary_df.to_string(index=False))
    print(f'Compression Ratio: {X_train_full_scaled_fs.nbytes / core_set_size:.2f}')

    # Visaulize lag and difference features, 



    # =========================
    # Step 10: Additional Evaluations (Optional)
    # =========================
    # For example, feature importance analysis, error analysis, etc.
    # You can add these steps as needed.

if __name__ == '__main__':
    main()

