# main.py

import os
import warnings
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Lasso

# Import custom classes
from data_loader import DataLoader
from dataPreprocessor import DataPreprocessor

# Import benchmarking functions
from benchmarking import (plot_learning_curve, evaluate_model,
                          perform_cross_validation, statistical_comparison, lasso_feature_selection)

import matplotlib.pyplot as plt
import seaborn as sns

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
    nodes = [f'node{i}' for i in range(8)] 
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


    # Check for NaN values after mapping
    unmapped_labels = data[data['Value'].isnull()]['Value']
    unmapped_labels_length = len(unmapped_labels)

    # Print unmapped labels and their length
    if unmapped_labels_length > 0:
        print(f"Unmapped labels after mapping (length={unmapped_labels_length}): {unmapped_labels}")
    else:
        print("No unmapped labels found.")

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

    # Split data indices before any preprocessing,
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
    #print("Feature columns after feature engineering:", feature_columns_fe)
    
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
    
    # Verify that all nodes are present after feature engineering
    feature_nodes_train = set([feature.split('.')[0] for feature in feature_columns_fe])
    print("Nodes represented in training features:", feature_nodes_train)
    
    feature_nodes_test = set([feature.split('.')[0] for feature in feature_columns_fe])
    
    # Record execution time
    end_time = time.time()
    execution_times['Feature Engineering and Normalization'] = end_time - start_time
    print(f"Feature engineering and normalization completed in {execution_times['Feature Engineering and Normalization']:.2f} seconds.")

    # ==========================================
    # Step 4: Feature Selection using Lasso
    # ==========================================
    print("\n=== Step 4: Feature Selection using Lasso ===")
    start_time = time.time()

    # Perform Lasso Feature Selection on training data
    X_train_full_scaled_fs, selected_features = lasso_feature_selection(X_train_full_scaled, y_train_full, alpha=0.001)
    print(f"Selected {len(selected_features)} features using Lasso.")

    # Apply the same feature selection to test data
    X_test_full_scaled_fs = X_test_full_scaled[:, selected_features]

    # Update feature columns
    feature_columns = [feature_columns_fe[i] for i in selected_features]

    # Check node representation in selected features
    selected_feature_nodes = set([feature.split('.')[0] for feature in feature_columns])
    print("Nodes represented in selected features:", selected_feature_nodes)

    # Record execution time
    end_time = time.time()
    execution_times['Feature Selection'] = end_time - start_time
    print(f"Feature selection completed in {execution_times['Feature Selection']:.2f} seconds.")

    # Update data size
    original_feature_count = X_train_full_scaled_fs.shape[1]
    original_data_size = X_train_full_scaled_fs.nbytes / 1024  # in KB
    print(f"Number of features after selection: {original_feature_count}")
    print(f"Original data size: {original_data_size:.2f} KB")

    # ==========================================
    # Step 5: Training Random Forest on Full Data
    # ==========================================
    start_time = time.time()
    print("\n=== Step 5: Training Random Forest on Full Data ===")

    # Train Random Forest on full training data
    rf_full = RandomForestClassifier(n_jobs=-1, random_state=42)
    rf_full.fit(X_train_full_scaled_fs, y_train_full)

    # Evaluate the full model on the full test data
    metrics_full = evaluate_model(
        rf_full, X_test_full_scaled_fs, y_test_full, class_names,
        model_name="Random Forest (Full Data)",
        filename_prefix="rf_full_data"
    )
    accuracy_full = metrics_full['accuracy']

    end_time = time.time()
    execution_times['Random Forest (Full Data)'] = end_time - start_time
    print(f"Random Forest trained and evaluated in {execution_times['Random Forest (Full Data)']:.2f} seconds.")

    # ==========================================
    # Step 5.1: Feature Importance Analysis
    # ==========================================
    print("\n=== Step 5.1: Feature Importance Analysis ===")

    # Get feature importances from the model
    importances = rf_full.feature_importances_

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

    # Cross-validation on full data
    print("\n=== Cross-Validation on Full Data ===")
    #skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    #cv_scores_full = cross_val_score(rf_full, X_train_full_scaled_fs, y_train_full, cv=skf, scoring='accuracy', n_jobs=-1)
    #print(f"Cross-validation scores on full data: {cv_scores_full}")
    #print(f"Mean accuracy on full data: {np.mean(cv_scores_full):.4f} +/- {np.std(cv_scores_full):.4f}")

    # ==================================================
    # Step 6: Selecting Core Set based on Stratified Sampling
    # ==================================================
    start_time = time.time()
    print("\n=== Step 6: Selecting Core Set based on Stratified Sampling ===")

    compression_ratio = 0.01  # Adjust as needed

    # Select the core set from the training data only
    # Note: We need to pass the data corresponding to the selected features
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

    rf_core = RandomForestClassifier(n_jobs=-1, random_state=42)
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

    # Cross-validation on core data
    #print("\n=== Cross-Validation on Core Data ===")
    #cv_scores_core = cross_val_score(rf_core, X_core, y_core, cv=skf, scoring='accuracy', n_jobs=-1)
    #print(f"Cross-validation scores on core data: {cv_scores_core}")
    #print(f"Mean accuracy on core data: {np.mean(cv_scores_core):.4f} +/- {np.std(cv_scores_core):.4f}")

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

    # Learning Curve Analysis for Core Model
    #print("\n=== Learning Curve Analysis for Core Data Model ===")
    #plot_learning_curve(rf_core, X_core, y_core, title="Learning Curve - Core Data", cv=5, n_jobs=-1)

    # =========================
    # Step 9: Statistical Comparison and Summary of Results
    # =========================
    print("\n=== Step 9: Statistical Comparison and Summary of Results ===")

    # Statistical comparison of cross-validation scores
    #statistical_comparison(cv_scores_full, cv_scores_core)

    # Calculate feature counts
    full_data_feature_count = X_train_full_scaled_fs.shape[1]

    core_data_feature_count = X_core.shape[1]

    summary_df = pd.DataFrame({
        'Dataset': ['Full Data', 'Core Set', 'Core Model on Full Test Data'],
        'Samples': [X_train_full_scaled_fs.shape[0], X_core.shape[0], X_test_full_scaled_fs.shape[0]],
        'Accuracy': [metrics_full['accuracy'], metrics_core['accuracy'], metrics_full_core['accuracy']],
        'Precision': [metrics_full['precision'], metrics_core['precision'], metrics_full_core['precision']],
        'Recall': [metrics_full['recall'], metrics_core['recall'], metrics_full_core['recall']],
        'F1 Score': [metrics_full['f1_score'], metrics_core['f1_score'], metrics_full_core['f1_score']],
        'Data Size (KB)': [original_data_size, core_set_size, X_test_full_scaled_fs.nbytes / 1024],
        'Number of Features': [full_data_feature_count, core_data_feature_count, full_data_feature_count]
    })

    print(summary_df.to_string(index=False))
    print(f'Compression Ratio: {original_data_size / core_set_size:.2f}')

    # =========================
    # Step 10: Additional Evaluations (Optional)
    # =========================
    # For example, feature importance analysis, error analysis, etc.
    # You can add these steps as needed.

if __name__ == '__main__':
    main()
