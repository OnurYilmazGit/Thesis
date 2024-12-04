import os
import warnings
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
from sklearn.manifold import TSNE

# Import custom classes
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from visualization import Visualization
from benchmark import Benchmark

# Suppress warnings
warnings.filterwarnings("ignore")


def example_usage(X_scaled, X_core, y_scaled, y_core, class_names):
    print("\n=== Benchmarking the Real and Core (Synthetic) Datasets ===")

    benchmark = Benchmark(X_scaled, X_core, y_scaled, y_core, class_names)

    # Step 1: Jensen-Shannon Divergence
    print("\n=== Step 1: Jensen-Shannon Divergence ===")
    benchmark.jensen_shannon_divergence()

    # Step 2: Statistical Comparison (Mean and Variance)
    print("\n=== Step 2: Statistical Comparison ===")
    benchmark.compare_statistics()


def plot_confusion_matrix(conf_matrix, class_names, title, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main():
    execution_times = {}

    visualization = Visualization()

    class_names = ['Kripke', 'AMG', 'PENNANT', 'linpack', 'LAMMPS', 'Quicksilver', 'None']

    # Step 1: Data Loading and Merging
    start_time = time.time()
    print("\n=== Step 1: Data Loading and Merging ===")

    responses_path = '../responses'
    sensors_path = '../sensors'
    nodes = [f'node{i}' for i in range(4)]

    data_loader = DataLoader(responses_path, sensors_path, nodes)
    responses = data_loader.load_responses()
    sensor_data = data_loader.load_sensors()

    label_mapping = {label: idx for idx, label in enumerate(class_names)}

    data = pd.merge(sensor_data, responses, on=['Time', 'Node'])
    data['Value'] = data['Value'].map(label_mapping)

    end_time = time.time()
    execution_times['Data Loading and Merging'] = end_time - start_time
    print(f"Data loaded and merged in {execution_times['Data Loading and Merging']:.2f} seconds.")
    print(f"Total records: {data.shape[0]}")

    # Step 2: Feature Engineering and Normalization
    start_time = time.time()
    print("\n=== Step 2: Feature Engineering and Normalization ===")

    preprocessor = DataPreprocessor(data, window_size=20)
    data = preprocessor.feature_engineering()

    if data.empty:
        print("Error: Data is empty after feature engineering.")
        return

    feature_columns = [col for col in data.columns if col not in ['Time', 'Node', 'Value', 'Cluster']]
    X = data[feature_columns]
    y = data['Value']

    X_scaled = preprocessor.normalize_data(feature_columns)

    original_feature_count = X_scaled.shape[1]
    original_data_size = X_scaled.nbytes / 1024  # in KB

    end_time = time.time()
    execution_times['Feature Engineering and Normalization'] = end_time - start_time
    print(f"Feature engineering and normalization completed in {execution_times['Feature Engineering and Normalization']:.2f} seconds.")

    # Step 3: Training Random Forest on Full Data
    start_time = time.time()
    print("\n=== Step 3: Training Random Forest on Full Data ===")

    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    rf_full = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42)
    rf_full.fit(X_train_full, y_train_full)

    y_pred_full = rf_full.predict(X_test_full)
    accuracy_full = accuracy_score(y_test_full, y_pred_full)

    conf_matrix_full = confusion_matrix(y_test_full, y_pred_full)
    plot_confusion_matrix(conf_matrix_full, class_names, "Confusion Matrix - Random Forest (Full Data)", "confusion_matrix_full.png")

    end_time = time.time()
    execution_times['Random Forest (Full Data)'] = end_time - start_time
    print(f"Random Forest trained on full data in {execution_times['Random Forest (Full Data)']:.2f} seconds.")
    print(f"Accuracy on full test set: {accuracy_full:.4f}")

    # Step 4: K-Means Clustering for Core Set Selection
    print("\n=== Step 4: K-Means Clustering for Core Set Selection ===")

    optimal_k = 500  # Increase clusters for better representation
    core_set = preprocessor.refine_cluster_selection(X_scaled, n_clusters=optimal_k, points_per_cluster=25)  # Increase points per cluster

    X_core = X_scaled[core_set.index]
    y_core = core_set['Value']

    core_set_size = X_core.nbytes / 1024  # in KB
    reduction_factor = X_scaled.shape[0] / X_core.shape[0]

    end_time = time.time()
    execution_times['Core Set Selection'] = end_time - start_time
    print(f"Core set selection completed in {execution_times['Core Set Selection']:.2f} seconds.")
    print(f"Core set size: {X_core.shape[0]} samples ({core_set_size:.2f} KB), Reduction factor: {reduction_factor:.2f}")

    # Step 5: Training Random Forest on Core Set Data
    start_time = time.time()
    print("\n=== Step 5: Training Random Forest on Core Set Data ===")

    X_train_core, X_test_core, y_train_core, y_test_core = train_test_split(X_core, y_core, test_size=0.2, random_state=42)

    rf_core = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42)
    rf_core.fit(X_train_core, y_train_core)

    y_pred_core = rf_core.predict(X_test_core)
    accuracy_core = accuracy_score(y_test_core, y_pred_core)

    conf_matrix_core = confusion_matrix(y_test_core, y_pred_core)
    plot_confusion_matrix(conf_matrix_core, class_names, "Confusion Matrix - Random Forest (Core Set)", "confusion_matrix_core.png")

    end_time = time.time()
    execution_times['Random Forest (Core Set)'] = end_time - start_time
    print(f"Random Forest trained on core set in {execution_times['Random Forest (Core Set)']:.2f} seconds.")
    print(f"Accuracy on core set test data: {accuracy_core:.4f}")

    # Step 6: Evaluating Core Set Model on Full Data
    print("\n=== Step 6: Evaluating Core Set Model on Full Data ===")

    y_pred_full_core = rf_core.predict(X_scaled)
    accuracy_full_core = accuracy_score(y, y_pred_full_core)

    conf_matrix_full_core = confusion_matrix(y, y_pred_full_core)
    plot_confusion_matrix(conf_matrix_full_core, class_names, "Confusion Matrix - Core Set Model on Full Data", "confusion_matrix_full_core.png")

    print(f"Accuracy of core set model on full dataset: {accuracy_full_core:.4f}")

    # Step 7: Summary of Results
    print("\n=== Step 7: Summary of Results ===")

    summary_df = pd.DataFrame({
        'Dataset': ['Full Data', 'Core Set', 'Core Model on Full Data'],
        'Samples': [X_scaled.shape[0], X_core.shape[0], X_scaled.shape[0]],
        'Accuracy': [accuracy_full, accuracy_core, accuracy_full_core],
        'Data Size (KB)': [original_data_size, core_set_size, original_data_size],
        'Number of Features': [X_scaled.shape[1], X_core.shape[1], X_scaled.shape[1]]
    })

    print(summary_df.to_string(index=False))
    print(f'Compression Ratio: {original_data_size / core_set_size}')

    # Benchmark Usage
    example_usage(X_scaled, X_core, y, y_core, class_names)

if __name__ == '__main__':
    main()