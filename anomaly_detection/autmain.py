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
from Autoencoder import Autoencoder

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

    
def main():
    # Initialize variables to store execution times
    execution_times = {}

    # Initialize visualization object
    visualization = Visualization()

    # Define class names, including 'none' for idle periods
    class_names = ['Kripke', 'AMG', 'PENNANT', 'linpack', 'LAMMPS', 'Quicksilver', 'None']

    # ================================
    # Step 1: Data Loading and Merging
    # ================================
    start_time = time.time()
    print("\n=== Step 1: Data Loading and Merging ===")
    
    responses_path = '../responses'
    sensors_path = '../sensors'
    nodes = [f'node{i}' for i in range(16)]  # Adjusted node count
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

    # ==============================================
    # Step 3: Applying Autoencoder for Compression
    # ==============================================
    print("\n=== Step 3: Applying Autoencoder for Compression ===")
    
    input_dim = X_scaled.shape[1]
    encoding_dim = 50  # You can adjust this based on how much compression you want
    
    autoencoder = Autoencoder(input_dim, encoding_dim)
    
    # Train the autoencoder
    autoencoder.train(X_scaled, epochs=50, batch_size=256)
    
    # Use the encoder to compress the data
    X_encoded = autoencoder.encode(X_scaled)
    print(f"Shape of Encoded Data: {X_encoded.shape}")
    
    # ==========================================
    # Step 4: K-Means Clustering for Core Set Selection on Encoded Data
    # ==========================================
    print("\n=== Step 4: K-Means Clustering for Core Set Selection on Encoded Data ===")
    
    optimal_k = 300
    print(f"Optimal number of clusters determined: {optimal_k}")

    core_set = preprocessor.refine_cluster_selection(X_encoded, n_clusters=optimal_k, points_per_cluster=8)

    X_core = X_encoded[core_set.index]
    y_core = core_set['Value']

    core_set_size = X_core.nbytes / 1024  # in KB
    reduction_factor = X_scaled.shape[0] / X_core.shape[0]

    end_time = time.time()
    execution_times['Core Set Selection'] = end_time - start_time
    print(f"Core set selection completed in {execution_times['Core Set Selection']:.2f} seconds.")
    print(f"Core set size: {X_core.shape[0]} samples ({core_set_size:.2f} KB), Reduction factor: {reduction_factor:.2f}")

    # =============================================
    # Step 5: Training Random Forest on Core Set Data
    # =============================================
    start_time = time.time()
    print("\n=== Step 5: Training Random Forest on Core Set Data ===")

    X_train_core, X_test_core, y_train_core, y_test_core = train_test_split(X_core, y_core, test_size=0.2, random_state=42)

    rf_core = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    rf_core.fit(X_train_core, y_train_core)

    y_pred_core = rf_core.predict(X_test_core)
    accuracy_core = accuracy_score(y_test_core, y_pred_core)

    conf_matrix_core = confusion_matrix(y_test_core, y_pred_core)
    plot_confusion_matrix(conf_matrix_core, class_names, "Confusion Matrix - Random Forest (Core Set)", "confusion_matrix_core.png")

    end_time = time.time()
    execution_times['Random Forest (Core Set)'] = end_time - start_time
    print(f"Random Forest trained and evaluated on core set in {execution_times['Random Forest (Core Set)']:.2f} seconds.")
    print(f"Accuracy on core set test data: {accuracy_core:.4f}")

    # =============================================
    # Step 6: Evaluating Core Set Model on Full Data
    # =============================================
    print("\n=== Step 6: Evaluating Core Set Model on Full Data ===")

    # Predict using the full dataset with the core set model
    y_pred_full_core = rf_core.predict(X_encoded)  # Use the full encoded dataset for prediction
    accuracy_full_core = accuracy_score(y, y_pred_full_core)

    conf_matrix_full_core = confusion_matrix(y, y_pred_full_core)
    plot_confusion_matrix(conf_matrix_full_core, class_names, "Confusion Matrix - Core Set Model on Full Data", "confusion_matrix_full_core.png")

    print(f"Accuracy of core set model on full dataset: {accuracy_full_core:.4f}")

    # =========================
    # Step 7: Summary of Results
    # =========================
    print("\n=== Step 7: Summary of Results ===")
    summary_df = pd.DataFrame({
        'Dataset': ['Full Data', 'Core Set', 'Core Model on Full Data'],
        'Samples': [X_scaled.shape[0], X_core.shape[0], X_scaled.shape[0]],
        'Accuracy': [accuracy_full, accuracy_core, accuracy_full_core],
        'Data Size (KB)': [original_data_size, core_set_size, original_data_size],
    })
    
    print(summary_df.to_string(index=False))
    print('Compression Ratio as', original_data_size / core_set_size)

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