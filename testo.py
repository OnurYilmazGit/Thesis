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

# Import custom classes
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from visualization import Visualization

# Suppress warnings
warnings.filterwarnings("ignore")

def plot_confusion_matrix(conf_matrix, class_names, title, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
def roll_forward_partitioning_core_set(X_core, y_core, pca_model, X_scaled, y, train_window_size=500, test_window_size=100):
    """
    Perform roll-forward partitioning on both core set and original data.
    """
    print("\n=== Step 9: Roll-Forward Partitioning on Core Set Data ===")
    
    # Core Set Roll-Forward
    num_samples_core = X_core.shape[0]
    avg_accuracy_core = 0
    partition_count_core = 0

    train_start = 0
    while train_start + train_window_size + test_window_size <= num_samples_core:
        train_end = train_start + train_window_size
        test_end = train_end + test_window_size

        X_train_core, X_test_core = X_core[train_start:train_end], X_core[train_end:test_end]
        y_train_core, y_test_core = y_core[train_start:train_end], y_core[train_end:test_end]

        rf_core = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
        rf_core.fit(X_train_core, y_train_core)

        y_pred_core = rf_core.predict(X_test_core)
        accuracy_core = accuracy_score(y_test_core, y_pred_core)
        avg_accuracy_core += accuracy_core
        partition_count_core += 1

        print(f"Core Set Partition: Train window: {train_start} to {train_end}, Test window: {train_end} to {test_end}, Accuracy: {accuracy_core:.4f}")

        train_start += test_window_size
    
    if partition_count_core > 0:
        avg_accuracy_core /= partition_count_core
        print(f"Average accuracy across roll-forward partitions (Core Set): {avg_accuracy_core:.4f}")
    else:
        print("Warning: No valid partitions were found for the core set.")
    
    # Now do the same for the original dataset
    print("\n=== Step 10: Roll-Forward Partitioning on Original Dataset ===")
    num_samples_original = X_scaled.shape[0]
    avg_accuracy_original = 0
    partition_count_original = 0

    train_start = 0
    while train_start + train_window_size + test_window_size <= num_samples_original:
        train_end = train_start + train_window_size
        test_end = train_end + test_window_size

        X_train_original, X_test_original = X_scaled[train_start:train_end], X_scaled[train_end:test_end]
        y_train_original, y_test_original = y[train_start:train_end], y[train_end:test_end]

        rf_original = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
        rf_original.fit(X_train_original, y_train_original)

        y_pred_original = rf_original.predict(X_test_original)
        accuracy_original = accuracy_score(y_test_original, y_pred_original)
        avg_accuracy_original += accuracy_original
        partition_count_original += 1

        print(f"Original Dataset Partition: Train window: {train_start} to {train_end}, Test window: {train_end} to {test_end}, Accuracy: {accuracy_original:.4f}")

        train_start += test_window_size
    
    if partition_count_original > 0:
        avg_accuracy_original /= partition_count_original
        print(f"Average accuracy across roll-forward partitions (Original Dataset): {avg_accuracy_original:.4f}")
    else:
        print("Warning: No valid partitions were found for the original dataset.")

    return avg_accuracy_core, avg_accuracy_original

def main():
    # Initialize variables to store execution times
    execution_times = {}

    # Initialize visualization object
    visualization = Visualization()

    # Define class names
    class_names = ['Kripke', 'AMG', 'PENNANT', 'linpack', 'LAMMPS', 'Quicksilver']

    # ================================
    # Step 1: Data Loading and Merging
    # ================================
    start_time = time.time()
    print("\n=== Step 1: Data Loading and Merging ===")
    
    responses_path = '../responses'
    sensors_path = '../sensors'
    nodes = [f'node{i}' for i in range(6)]  # Adjusted node count
    print(f"Length of Nodes: {len(nodes)}")

    data_loader = DataLoader(responses_path, sensors_path, nodes)
    responses = data_loader.load_responses()
    sensor_data = data_loader.load_sensors()
    
    data = pd.merge(sensor_data, responses, on=['Time', 'Node'])
    data['Value'] = data['Value'].map({label: idx for idx, label in enumerate(class_names)})
    
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

    feature_columns = [col for col in data.columns if col not in ['Time', 'Node', 'Value', 'Cluster']]
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

    # =========================================
    # Step 3: PCA for Dimensionality Reduction
    # =========================================
    start_time = time.time()
    print("\n=== Step 3: PCA for Dimensionality Reduction ===")
    
    X_pca, pca_model = preprocessor.apply_pca(X_scaled, n_components=None)
    
    cumulative_variance = np.cumsum(pca_model.explained_variance_ratio_)
    optimal_components = np.argmax(cumulative_variance >= 0.16) + 1  # +1 because indices start at 0
    
    print(f"Optimal number of PCA components to retain 30% variance: {optimal_components}")
    
    # Re-apply PCA with optimal components
    X_pca, pca_model = preprocessor.apply_pca(X_scaled, n_components=optimal_components)

    reduced_data_size = X_pca.nbytes / 1024  # in KB

    end_time = time.time()
    execution_times['PCA'] = end_time - start_time
    print(f"PCA completed in {execution_times['PCA']:.2f} seconds.")
    print(f"Reduced data size after PCA: {reduced_data_size:.2f} KB")

    # ==========================================
    # Step 4: Training Random Forest on Full PCA-Reduced Data
    # ==========================================
    start_time = time.time()
    print("\n=== Step 4: Training Random Forest on Full PCA-Reduced Data ===")

    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    rf_full = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
    rf_full.fit(X_train_full, y_train_full)

    y_pred_full = rf_full.predict(X_test_full)
    accuracy_full = accuracy_score(y_test_full, y_pred_full)

    conf_matrix_full = confusion_matrix(y_test_full, y_pred_full)

    end_time = time.time()
    execution_times['Random Forest (Full Data)'] = end_time - start_time
    print(f"Random Forest trained and evaluated in {execution_times['Random Forest (Full Data)']:.2f} seconds.")
    print(f"Accuracy on full PCA-reduced test set: {accuracy_full:.4f}")

    # ==============================================
    # Step 5: K-Means Clustering for Core Set Selection
    # ==============================================
    print("\n=== Step 5: K-Means Clustering for Core Set Selection ===")

    optimal_k = 150
    print(f"Optimal number of clusters determined: {optimal_k}")

    core_set = preprocessor.refine_cluster_selection(X_pca, n_clusters=optimal_k, points_per_cluster=15)

    X_core = X_pca[core_set.index]  
    y_core = core_set['Value']

    core_set_size = X_core.nbytes / 1024  # in KB
    reduction_factor = X_pca.shape[0] / X_core.shape[0]

    end_time = time.time()
    execution_times['Core Set Selection'] = end_time - start_time
    print(f"Core set selection completed in {execution_times['Core Set Selection']:.2f} seconds.")
    print(f"Core set size: {X_core.shape[0]} samples ({core_set_size:.2f} KB), Reduction factor: {reduction_factor:.2f}")

    # =============================================
    # Step 6: Training Random Forest on Core Set Data
    # =============================================
    start_time = time.time()
    print("\n=== Step 6: Training Random Forest on Core Set Data ===")

    X_train_core, X_test_core, y_train_core, y_test_core = train_test_split(X_core, y_core, test_size=0.2, random_state=42)

    rf_core = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    rf_core.fit(X_train_core, y_train_core)

    y_pred_core = rf_core.predict(X_test_core)
    accuracy_core = accuracy_score(y_test_core, y_pred_core)

    conf_matrix_core = confusion_matrix(y_test_core, y_pred_core)

    end_time = time.time()
    execution_times['Random Forest (Core Set)'] = end_time - start_time
    print(f"Random Forest trained and evaluated on core set in {execution_times['Random Forest (Core Set)']:.2f} seconds.")
    print(f"Accuracy on core set test data: {accuracy_core:.4f}")   

    # =============================================
    # Step 7: Roll-Forward Partitioning on Core Set and Original Dataset
    # =============================================
    avg_accuracy_core, avg_accuracy_original = roll_forward_partitioning_core_set(X_core, y_core, pca_model, X_scaled, y)

    print(f"Roll-Forward Partitioning - Core Set Average Accuracy: {avg_accuracy_core:.4f}")
    print(f"Roll-Forward Partitioning - Original Dataset Average Accuracy: {avg_accuracy_original:.4f}")

    # =========================
    # Summary of Results
    # =========================
    print("\n=== Step 8: Summary of Results ===")

    compression_ratio_pca = original_data_size / reduced_data_size
    compression_ratio_core = original_data_size / core_set_size

    summary_df = pd.DataFrame({
        'Dataset': ['Full Data', 'Core Set', 'Roll-Forward Core Set', 'Roll-Forward Original'],
        'Samples': [X_pca.shape[0], X_core.shape[0], X_pca.shape[0], X_scaled.shape[0]],
        'Features': [X_pca.shape[1], X_core.shape[1], X_pca.shape[1], X_scaled.shape[1]],
        'Accuracy': [accuracy_full, accuracy_core, avg_accuracy_core, avg_accuracy_original],
        'Data Size (KB)': [reduced_data_size, core_set_size, reduced_data_size, original_data_size],
        'Compression Ratio': [compression_ratio_pca, compression_ratio_core, 'N/A', 'N/A']
    })

    print(summary_df.to_string(index=False))


if __name__ == '__main__':
    main()

