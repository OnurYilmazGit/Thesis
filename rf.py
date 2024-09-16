# main.py

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
from sklearn.manifold import TSNE  # Import for t-SNE
from sklearn.metrics import precision_score, recall_score, f1_score

# Import custom classes
from data_loader import DataLoader
from dataPreprocessor import DataPreprocessor
from visualization import Visualization

# Suppress warnings
warnings.filterwarnings("ignore")

from benchmark import Benchmark  # Assuming you have saved the Benchmark class as benchmark.py

def example_usage(X_scaled, X_core, y_scaled, y_core, class_names):
    """
    Example usage of the Benchmark class for comparing real and synthetic datasets.
    
    Args:
        X_scaled (ndarray): The real (full) dataset.
        X_core (ndarray): The synthetic (core) dataset.
        y_scaled (ndarray): Labels for the real (full) dataset.
        y_core (ndarray): Labels for the synthetic (core) dataset.
        class_names (list): List of class names.
    """
    print("\n=== Benchmarking the Real and Core (Synthetic) Datasets ===")

    # Initialize the benchmark class
    benchmark = Benchmark(X_scaled, X_core, y_scaled, y_core, class_names)

    # Step 1: Jensen-Shannon Divergence
    print("\n=== Step 1: Jensen-Shannon Divergence ===")
    benchmark.jensen_shannon_divergence()

    # Step 2: Statistical Comparison (Mean and Variance)
    print("\n=== Step 2: Statistical Comparison ===")
    benchmark.compare_statistics()
    
    # Optional: you can also call the t-SNE visualization method here if needed:
    #benchmark.tsne_visualization()

def plot_tsne(X_full, X_core, y_full, y_core, class_names):
    tsne = TSNE(n_components=2, random_state=42)

    # t-SNE for full dataset
    X_full_tsne = tsne.fit_transform(X_full)

    # t-SNE for core dataset
    X_core_tsne = tsne.fit_transform(X_core)

    plt.figure(figsize=(12, 6))

    # Plot full dataset
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_full_tsne[:, 0], X_full_tsne[:, 1], c=y_full, cmap='viridis', s=2)
    plt.colorbar(scatter, ticks=range(len(class_names)))
    plt.title("t-SNE of Full Dataset")

    # Plot core dataset
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_core_tsne[:, 0], X_core_tsne[:, 1], c=y_core, cmap='viridis', s=2)
    plt.colorbar(scatter, ticks=range(len(class_names)))
    plt.title("t-SNE of Core Set")

    plt.tight_layout()
    plt.show()

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
    # Initialize variables to store execution times
    execution_times = {}

    # Initialize visualization object
    visualization = Visualization()

    # Define class names, including 'None' for idle periods
    class_names = ['Kripke', 'AMG', 'PENNANT', 'linpack', 'LAMMPS', 'Quicksilver', 'None']

    # ================================
    # Step 1: Data Loading and Merging
    # ================================
    start_time = time.time()
    print("\n=== Step 1: Data Loading and Merging ===")
    
    responses_path = '../responses'
    sensors_path = '../sensors'
    nodes = [f'node{i}' for i in range(12)]  # Adjusted node count
    print(f"Length of Nodes: {len(nodes)}")

    data_loader = DataLoader(responses_path, sensors_path, nodes)
    responses = data_loader.load_responses()
    sensor_data = data_loader.load_sensors()

    # Map application labels, including 'None' for idle periods
    label_mapping = {label: idx for idx, label in enumerate(class_names)}

    # Merge data and map labels
    data = pd.merge(sensor_data, responses, on=['Time', 'Node'])
    data['Value'] = data['Value'].map(label_mapping)
    
    end_time = time.time()
    execution_times['Data Loading and Merging'] = end_time - start_time
    print(f"Data loaded and merged successfully in {execution_times['Data Loading and Merging']:.2f} seconds.")
    print(f"Total records: {data.shape[0]}")

    # ============================================
    # Step 2: Feature Engineering and Normalization
    # ============================================
    start_time = time.time()
    print("\n=== Step 2: Feature Engineering and Normalization ===")

    preprocessor = DataPreprocessor(data, window_size=20)
    data = preprocessor.feature_engineering()

    # Check if data is empty after feature engineering
    if data.empty:
        print("Error: Data is empty after feature engineering. Saving snapshot for debugging and exiting.")
        data.to_csv('empty_data_snapshot.csv', index=False)
        return

    feature_columns = [col for col in data.columns if col not in ['Time', 'Node', 'Value']]
    X = data[feature_columns]
    y = data['Value']

    X_scaled = preprocessor.normalize_data(feature_columns)

    original_feature_count = X_scaled.shape[1]
    original_data_size = X_scaled.nbytes / 1024  # in KB

    end_time = time.time()
    execution_times['Feature Engineering and Normalization'] = end_time - start_time
    print(f"Feature engineering and normalization completed in {execution_times['Feature Engineering and Normalization']:.2f} seconds.")
    print(f"Original number of features: {original_feature_count}")
    print(f"Original data size: {original_data_size:.2f} KB")

    # ==========================================
    # Step 3: Training Random Forest on Full Data with Additional Metrics
    # ==========================================
    start_time = time.time()
    print("\n=== Step 3: Training Random Forest on Full Data ===")

    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    rf_full = RandomForestClassifier(n_jobs=-1, random_state=42)
    rf_full.fit(X_train_full, y_train_full)

    y_pred_full = rf_full.predict(X_test_full)

    # Calculate additional metrics
    accuracy_full = accuracy_score(y_test_full, y_pred_full)
    precision_full = precision_score(y_test_full, y_pred_full, average='weighted')
    recall_full = recall_score(y_test_full, y_pred_full, average='weighted')
    f1_full = f1_score(y_test_full, y_pred_full, average='weighted')

    # Print additional metrics
    print(f"Accuracy on full test set: {accuracy_full:.4f}")
    print(f"Precision on full test set: {precision_full:.4f}")
    print(f"Recall on full test set: {recall_full:.4f}")
    print(f"F1 Score on full test set: {f1_full:.4f}")

    # Confusion Matrix
    conf_matrix_full = confusion_matrix(y_test_full, y_pred_full)
    plot_confusion_matrix(conf_matrix_full, class_names, "Confusion Matrix - Random Forest (Full Data)", "confusion_matrix_full.png")

    end_time = time.time()
    execution_times['Random Forest (Full Data)'] = end_time - start_time
    print(f"Random Forest trained and evaluated in {execution_times['Random Forest (Full Data)']:.2f} seconds.")

    # ==================================================
    # Step 4: Selecting Core Set based on Random Forest Importance Scores
    # ==================================================
    start_time = time.time()
    print("\n=== Step 4: Selecting Core Set based on Random Forest Importance Scores ===")

    compression_ratio = 0.1  # Adjust as needed

    core_set = preprocessor.select_core_set_by_rf(X_scaled, y, compression_ratio=compression_ratio)

    # Ensure indices align correctly
    X_core = X_scaled[core_set.index.to_numpy()]
    y_core = y.iloc[core_set.index]

    core_set_size = X_core.nbytes / 1024  # in KB
    reduction_factor = X_scaled.shape[0] / X_core.shape[0]

    end_time = time.time()
    execution_times['Core Set Selection'] = end_time - start_time
    print(f"Core set selection completed in {execution_times['Core Set Selection']:.2f} seconds.")
    print(f"Core set size: {X_core.shape[0]} samples ({core_set_size:.2f} KB), Reduction factor: {reduction_factor:.2f}")

    # =============================================
    # Step 5: Training Random Forest on Core Set Data with Additional Metrics
    # =============================================
    start_time = time.time()
    print("\n=== Step 5: Training Random Forest on Core Set Data ===")

    X_train_core, X_test_core, y_train_core, y_test_core = train_test_split(X_core, y_core, test_size=0.2, random_state=42)

    rf_core = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    rf_core.fit(X_train_core, y_train_core)

    y_pred_core = rf_core.predict(X_test_core)

    # Calculate additional metrics
    accuracy_core = accuracy_score(y_test_core, y_pred_core)
    precision_core = precision_score(y_test_core, y_pred_core, average='weighted')
    recall_core = recall_score(y_test_core, y_pred_core, average='weighted')
    f1_core = f1_score(y_test_core, y_pred_core, average='weighted')

    # Print additional metrics
    print(f"Accuracy on core set test data: {accuracy_core:.4f}")
    print(f"Precision on core set test data: {precision_core:.4f}")
    print(f"Recall on core set test data: {recall_core:.4f}")
    print(f"F1 Score on core set test data: {f1_core:.4f}")

    # Confusion Matrix
    conf_matrix_core = confusion_matrix(y_test_core, y_pred_core)
    plot_confusion_matrix(conf_matrix_core, class_names, "Confusion Matrix - Random Forest (Core Set)", "confusion_matrix_core.png")

    end_time = time.time()
    execution_times['Random Forest (Core Set)'] = end_time - start_time
    print(f"Random Forest trained and evaluated on core set in {execution_times['Random Forest (Core Set)']:.2f} seconds.")

    # =============================================
    # Step 6: Evaluating Core Set Model on Full Data
    # =============================================
    print("\n=== Step 6: Evaluating Core Set Model on Full Data ===")

    # Predict using the full dataset with the core set model
    y_pred_full_core = rf_core.predict(X_scaled)  # Use the full dataset (X_scaled) for prediction
    accuracy_full_core = accuracy_score(y, y_pred_full_core)  # Compare with original full labels (y)

    conf_matrix_full_core = confusion_matrix(y, y_pred_full_core)
    plot_confusion_matrix(conf_matrix_full_core, class_names, "Confusion Matrix - Core Set Model on Full Data", "confusion_matrix_full_core.png")

    print(f"Accuracy of core set model on full dataset: {accuracy_full_core:.4f}")

    # =========================
    # Step 7: Summary of Results
    # =========================
    print("\n=== Step 7: Summary of Results ===")
    
    # Calculate feature counts
    full_data_feature_count = X_scaled.shape[1]
    core_data_feature_count = X_core.shape[1]
    
    summary_df = pd.DataFrame({
        'Dataset': ['Full Data', 'Core Set', 'Core Model on Full Data'],
        'Samples': [X_scaled.shape[0], X_core.shape[0], X_scaled.shape[0]],
        'Accuracy': [accuracy_full, accuracy_core, accuracy_full_core],
        'Data Size (KB)': [original_data_size, core_set_size, original_data_size],
        'Number of Features': [full_data_feature_count, core_data_feature_count, full_data_feature_count]
    })
    
    print(summary_df.to_string(index=False))
    print('Compression Ratio as', original_data_size / core_set_size)

    # ========================= Benchmark Class Usage =========================
    #example_usage(X_scaled, X_core, y, y_core, class_names)

    # ========================= t-SNE Visualization =========================
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    plot_files = ['confusion_matrix_full.png', 'confusion_matrix_core.png', 'confusion_matrix_full_core.png']
    titles = ['Random Forest (Full Data)', 'Random Forest (Core Set)', 'Core Set Model on Full Data']

    for ax, file, title in zip(axes, plot_files, titles):
        img = plt.imread(file)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(title)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
