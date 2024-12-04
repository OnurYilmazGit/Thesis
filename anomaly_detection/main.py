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
    nodes = [f'node{i}' for i in range(12)]  # Adjusted node count
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
    print(cumulative_variance)
    
    print(f"Optimal number of PCA components to retain 30% variance: {optimal_components}")
    
    # Re-apply PCA with optimal components
    X_pca, pca_model = preprocessor.apply_pca(X_scaled, n_components= optimal_components)

    reduced_data_size = X_pca.nbytes / 1024  # in KB

    end_time = time.time()
    execution_times['PCA'] = end_time - start_time
    print(f"PCA completed in {execution_times['PCA']:.2f} seconds.")
    print(f"Reduced data size after PCA: {reduced_data_size:.2f} KB")

    # ==========================================
    # Step 4: Training Random Forest on Full Data
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
    # Step 7: Evaluating Core Set Model on Full PCA-Reduced Data
    # =============================================
    start_time = time.time()
    print("\n=== Step 7: Evaluating Core Set Model on Full PCA-Reduced Data ===")

    y_pred_full_core = rf_core.predict(X_pca)
    accuracy_full_core = accuracy_score(y, y_pred_full_core)

    conf_matrix_full_core = confusion_matrix(y, y_pred_full_core)

    end_time = time.time()
    execution_times['Evaluation on Full Data'] = end_time - start_time
    print(f"Evaluation completed in {execution_times['Evaluation on Full Data']:.2f} seconds.")
    print(f"Accuracy of core set model on full PCA-reduced data: {accuracy_full_core:.4f}")
    # =============================================
    # Step 8: Evaluating Core Set Model on Original Full Dataset (Before PCA)
    # =============================================
    start_time = time.time()
    print("\n=== Step 8: Evaluating Core Set Model on Original Full Dataset ===")

    # Apply the same PCA transformation to the original full dataset
    X_scaled_pca = pca_model.transform(X_scaled)

    # Now predict using the PCA-transformed original dataset
    y_pred_core_on_original = rf_core.predict(X_scaled_pca)
    accuracy_core_on_original = accuracy_score(y, y_pred_core_on_original)
    mse_core_on_original = mean_squared_error(y, y_pred_core_on_original)

    conf_matrix_core_on_original = confusion_matrix(y, y_pred_core_on_original)
    plot_confusion_matrix(conf_matrix_core_on_original, class_names, "Confusion Matrix - Core Set Model on Original Full Data", "confusion_matrix_core_on_original.png")

    end_time = time.time()
    execution_times['Core Model on Original Full Data'] = end_time - start_time
    print(f"Evaluation of core set model on original full dataset completed in {execution_times['Core Model on Original Full Data']:.2f} seconds.")
    print(f"Accuracy of core set model on original full dataset: {accuracy_core_on_original:.4f}")
    print(f"Mean Squared Error of core set model on original full dataset: {mse_core_on_original:.4f}")


    # =============================
    # Step 9: Visualizing Decomposed Components and Autocorrelation
    # =============================
    print("\n=== Step 9: Visualizing Decomposed Components and Autocorrelation ===")
    
    # Assuming X_pca[:, 0] is the first PCA component (you can repeat for other components)
    #visualization = Visualization()
    
    # Plot the seasonal decomposition for the first PCA component
    #visualization.plot_seasonal_decomposition(X_pca[:, 0], frequency=12, title="Seasonality, Trend, and Noise - PCA Component 1")
    
    # Plot the autocorrelation for the first PCA component
    #visualization.plot_autocorrelation(X_pca[:, 0], lags=50, title="Autocorrelation - PCA Component 1")
    
    # Plot the partial autocorrelation for the first PCA component
    #visualization.plot_partial_autocorrelation(X_pca[:, 0], lags=50, title="Partial Autocorrelation - PCA Component 1")
    
    # If you want to visualize more PCA components, you can repeat for X_pca[:, 1], X_pca[:, 2], etc.
    

    # =========================
    # Step 9: Summary of Results
    # =========================
    print("\n=== Step 9: Summary of Results ===")

    compression_ratio_pca = original_data_size / reduced_data_size
    compression_ratio_core = original_data_size / core_set_size

    summary_df = pd.DataFrame({
        'Dataset': ['Full Data', 'Core Set', 'Core Model on Full Data', 'Core Model on Original Data'],
        'Samples': [X_pca.shape[0], X_core.shape[0], X_pca.shape[0], X_scaled.shape[0]],
        'Features': [X_pca.shape[1], X_core.shape[1], X_pca.shape[1], X_scaled.shape[1]],
        'Accuracy': [accuracy_full, accuracy_core, accuracy_full_core, accuracy_core_on_original],
        'Mean Squared Error': [None, None, None, mse_core_on_original],
        'Data Size (KB)': [reduced_data_size, core_set_size, reduced_data_size, original_data_size],
        'Training Time (s)': [
            execution_times['Random Forest (Full Data)'],
            execution_times['Random Forest (Core Set)'],
            'N/A',
            'N/A'
        ],
        'Compression Ratio': [compression_ratio_pca, compression_ratio_core, 'N/A', 'N/A']
    })

    print(summary_df.to_string(index=False))

    # =========================
    # Display All Plots Together
    # =========================
    print("\n=== Displaying All Plots ===")

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
